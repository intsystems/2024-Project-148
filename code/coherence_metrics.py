import numpy as np
import pandas as pd

class NewmanCoherence:
    def _get_topics(self, model, skip_bcg_topics):
        return [t for t in model.get_phi().columns if 'topic' in t or not skip_bcg_topics]

    def _calculate_topic_coherence(self, topic, phi, ppmi_dict, k):
        def getTopWords(distribution, k):
            order = np.argsort(distribution)[::-1]
            return distribution[order][:k]
        
        distribution = phi[topic]
        topWords = getTopWords(distribution, k)
        
        cohSum = 0
        relSum = 0
        for i, (u, pu) in enumerate(topWords.items()):
            for j, (v, pv) in enumerate(topWords.items()):
                if i < j and u in ppmi_dict and  v in ppmi_dict[u]:
                    relSum += 1
                    cohSum += ppmi_dict[u][v]
        return cohSum / relSum
                

    def call(self, model, ppmi_handler, k=10, modality='@lemmatized', skip_bcg_topics=True):
        topics = self._get_topics(model, skip_bcg_topics)
        phi = model.get_phi().loc[modality]
        return np.array([self._calculate_topic_coherence(t, phi, ppmi_handler.ppmi, k) for t in topics])

class CustomRelCoherence:
    def __init__(self, rel_function):
        self.rel = rel_function

    def _get_topics(self, model, skip_bcg_topics):
        return [t for t in model.get_phi().columns if 'topic' in t or not skip_bcg_topics]

    def _calculate_topic_coherence(self, topic, phi, ppmi_dict):
        def orderDistribution(distribution):
            order = np.argsort(distribution)[::-1]
            return distribution[order]
        distribution = orderDistribution(phi[topic])
        relSum = 0
        cohSum = 0
        for i, (u, pu) in enumerate(distribution.items()):
            for j, (v, pv) in enumerate(distribution.items()):
                if pu > 0 and pv > 0 and self.rel(pu, pv) > 0:
                    if i < j and u in ppmi_dict and v in ppmi_dict[u]:
                        relSum += self.rel(pu, pv)
                        cohSum += self.rel(pu, pv) * ppmi_dict[u][v]
                else:
                    break
        return cohSum / relSum

    def call(self, model, ppmi_handler, modality='@lemmatized', skip_bcg_topics=True):
        topics = self._get_topics(model, skip_bcg_topics)
        phi = model.get_phi().loc[modality]
        return np.array([self._calculate_topic_coherence(t, phi, ppmi_handler.ppmi) for t in topics])
