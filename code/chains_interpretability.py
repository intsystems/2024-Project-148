import numpy as np
import pandas as pd

class ChainsInterpretability:
    def __init__(self, chains):
        super().__init__()
        self.chains = chains

    def _get_topics(self, model):
        return list(model.get_phi().columns)
    
    def _calculate_interpretability(self, topics, phi):
        data = [[] for _ in range(len(topics))]
        for chain in self.chains:
            probs = phi.loc[chain].to_numpy()
            likelyhood = np.log(probs).sum(axis=0)
            if np.argmax(likelyhood) == 20:
                print(chain)
            data[np.argmax(likelyhood)].append(np.max(likelyhood))
        return pd.Series([np.mean(row) if row.size else -np.inf for row in data], index=topics)

    def call(self, model):
        topics = self._get_topics(model)
        phi = model.get_phi()
        result = self._calculate_interpretability(topics, phi)
        return list(result)