import pickle
import numpy as np

from torch import tensor
from hitl_al_gomg.synthitl.epig import epig_from_probs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestClf(RandomForestClassifier):
    def __init__(self, fitted_model):
        super(RandomForestClf, self).__init__()
        self.fitted_model = fitted_model

    def _estimators(self):
        return self.fitted_model.estimators_

    def _predict_proba(self, x):
        return self.fitted_model.predict_proba(x)

    def _retrain(self, x, y, sample_weight=None, save_to_path=None):
        init_params = self.fitted_model.get_params()
        rf = RandomForestClassifier(**init_params)
        rf.fit(x, y, sample_weight=sample_weight)

        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, "wb"))

    def _get_prob_distribution(self, x):
        prob_dist = [estimator.predict_proba(x) for estimator in self._estimators()]
        prob_dist = np.stack(prob_dist, axis=1)
        prob_dist = tensor(prob_dist)
        return prob_dist

    def _entropy(self, x):
        probabilities = self.fitted_model.predict_proba(x)
        class_entropies = []

        for i in range(len(probabilities)):
            class_entropies.append(
                # Shannon entropy
                -np.sum(
                    [
                        probabilities[i, 0] * np.log2(probabilities[i, 0]),
                        probabilities[i, 1] * np.log2(probabilities[i, 1]),
                    ]
                )
            )
        return class_entropies

    def _estimate_epig(self, prob_pool: tensor, prob_target: tensor):
        return epig_from_probs(prob_pool, prob_target)


class RandomForestReg(RandomForestRegressor):
    def __init__(self, fitted_model):
        super(RandomForestReg, self).__init__()
        self.fitted_model = fitted_model

    def _estimators(self):
        return self.fitted_model.estimators_

    def _predict(self, x):
        return self.fitted_model.predict(x)

    def _retrain(self, x, y, sample_weight=None, save_to_path=None):
        init_params = self.fitted_model.get_params()
        init_params["n_jobs"] = -1
        rf = RandomForestRegressor(**init_params)
        rf.fit(x, y, sample_weight=sample_weight)
        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, "wb"))

    def _uncertainty(self, x):
        individual_trees = self.fitted_model.estimators_
        subEstimates = np.array(
            [tree.predict(np.stack(x)) for tree in individual_trees]
        )
        return np.std(subEstimates, axis=0)

    def _get_prob_distribution(self, x):
        prob_dist = [estimator.predict(x) for estimator in self._estimators()]
        prob_dist = np.stack(prob_dist, axis=1)
        prob_dist = tensor(prob_dist)
        return prob_dist

    def _estimate_epig(self, prob_pool: tensor, prob_target: tensor):
        return epig_from_probs(prob_pool, prob_target, classification=False)
