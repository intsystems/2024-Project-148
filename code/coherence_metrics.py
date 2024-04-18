import numpy as np
import pandas as pd

class NewmanCoherence:
    def __init__(self, coh_function, k):
        self.k = k
        self.coh = coh_function

    def _get_topics(self, model, skip_bcg_topics):
        return [t for t in model.get_phi().columns if 'topic' in t or not skip_bcg_topics]

    def _calculate_topic_coherence(self, topic, phi):
        def getTopWords(distribution, k):
            order = np.argsort(distribution)[::-1]
            return distribution[order][:k]
        
        distribution = phi[topic]
        topWords = getTopWords(distribution, self.k)
        
        cohSum = 0
        relSum = 0
        for i, (u, pu) in enumerate(topWords.items()):
            for j, (v, pv) in enumerate(topWords.items()):
                if i < j:
                    relSum += 1
                    cohSum += 1 * self.coh(u, v)
        return cohSum / relSum
                

    def call(self, model, modality='@lemmatized', skip_bcg_topics=True):
        topics = self._get_topics(model, skip_bcg_topics)
        phi = model.get_phi().loc[modality]
        return np.array([self._calculate_topic_coherence(t, phi) for t in topics])

class CustomRelCoherence:
    def __init__(self, rel_function, coh_function):
        self.rel = rel_function
        self.coh = coh_function

    def _get_topics(self, model, skip_bcg_topics):
        return [t for t in model.get_phi().columns if 'topic' in t or not skip_bcg_topics]

    def _calculate_topic_coherence(self, topic, phi):
        def orderDistribution(distribution):
            order = np.argsort(distribution)[::-1]
            return distribution[order]
        distribution = list(orderDistribution(phi[topic]).items())
        relSum = 0
        cohSum = 0
        for i, (u, pu) in enumerate(distribution):
            valid_items = distribution[:i]
            for j, (v, pv) in enumerate(valid_items):
                if pu > 0 and pv > 0 and self.rel(pu, pv) > 0:
                    relSum += self.rel(pu, pv)
                    cohSum += self.rel(pu, pv) * self.coh(u, v)
                else:
                    break
        return cohSum / relSum

    def call(self, model, modality='@lemmatized', skip_bcg_topics=True):
        topics = self._get_topics(model, skip_bcg_topics)
        phi = model.get_phi().loc[modality]
        return np.array([self._calculate_topic_coherence(t, phi) for t in topics])
