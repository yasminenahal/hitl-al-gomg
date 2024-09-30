import pickle
import numpy as np

from tdc import Oracle
from hitl_al_gomg.utils import double_sigmoid, ecfp_generator


class EvaluationModel:
    def __init__(self, task, path_to_simulator=None):
        self.task = task
        if self.task == "logp":
            self.oracle = Oracle(name="LogP")
        else:
            try:
                self.oracle = pickle.load(open(f"{path_to_simulator}", "rb"))
            except:
                ValueError("Path to pickled simulator required.")

    def oracle_score(self, smi):
        if smi:
            if self.task == "logp":
                score = self.oracle(smi)
            else:
                fp_counter = ecfp_generator(radius=3, useCounts=True)
                score = self.oracle.predict_proba(fp_counter.get_fingerprints([smi]))[
                    :, 1
                ].item()
        else:
            score = 0
        return float(score)

    def human_score(self, smi, sigma):
        if smi:
            if sigma > 0:
                noise = np.random.normal(0, sigma, 1).item()
            else:
                noise = 0
            if self.task == "logp":
                expert_score = self.oracle_score(smi) + noise
                expert_util = utility(expert_score, low=2, high=4)
                print(f"human score : {expert_util} (approx. value = {expert_score})")
            else:
                expert_score = np.clip(self.oracle_score(smi) + noise, 0, 1)
                print(f"human score : {expert_score}")
            return float(expert_score)
        else:
            return 0.0


def utility(x, low, high, alpha_1=10, alpha_2=10):
    utility_score = double_sigmoid(x, low, high, alpha_1, alpha_2)
    return utility_score
