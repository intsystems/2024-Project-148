import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ChainsConsistency:
    def __init__(self, chains, modality='lemmatized'):
        self.chains = chains

    def _get_topics(self, model):
        return list(model.get_phi().columns)
    
    def _calculate_interpretability(self, topics, phi, n_w, n_t, draw_distribution):
        Ct = [[] for _ in range(len(topics))]
        for chain in self.chains:
            p_tw = (phi.loc[chain].to_numpy() / n_t.to_numpy()).T * n_w.loc[chain].to_numpy()
            p_tC = np.mean(p_tw, axis=-1)
            Ct[np.argmax(p_tC)].append(np.max(p_tC))

        if draw_distribution:
            plt.xlabel('Topics')
            plt.ylabel('Count of chains')
            plt.xticks(rotation=90)
            plt.bar(topics, [len(row) for row in Ct])
        return pd.Series([np.mean(row) if len(row) else -np.inf for row in Ct], index=topics)

    def visualize(self, model, modality='@lemmatized'):
        topics = self._get_topics(model)
        phi = model.get_phi().loc[modality]
        n_wt = model._model.get_phi(model_name=model._model.model_nwt)
        p_wt = model._model.get_phi(model_name=model._model.model_pwt)

        n_w = n_wt.sum(axis=1)
        n_t = np.mean((n_wt / p_wt).replace([np.inf, -np.inf], np.nan).dropna(),axis=0)
        

        plt.subplots(1, 2, figsize=(10, 3))
        plt.subplot(1, 2, 1)
        result = self._calculate_interpretability(topics, phi, n_w, n_t, True)

        plt.subplot(1, 2, 2)
        plt.xlabel('Topics')
        plt.ylabel(r'$cons_t$')
        plt.xticks(rotation=90)
        plt.bar(topics, [i if i != -np.inf else 0 for i in result])
        plt.show()


    def call(self, model, modality='@lemmatized'):
        topics = self._get_topics(model)
        phi = model.get_phi().loc[modality]
        n_wt = model._model.get_phi(model_name=model._model.model_nwt)
        p_wt = model._model.get_phi(model_name=model._model.model_pwt)

        n_w = n_wt.sum(axis=1)
        n_t = np.mean((n_wt / p_wt).replace([np.inf, -np.inf], np.nan).dropna(),axis=0)
        result = self._calculate_interpretability(topics, phi, n_w, n_t, False)
        return list(result)
