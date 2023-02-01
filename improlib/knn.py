import numpy as np


class KnnClassifier:
    def __init__(
        self,
        labels,
        samples
    ):
        self.labels = labels
        self.samples = samples
    
    def classify(
        self,
        point,
        k=3
    ):
        dist = np.array([l2_dist(point, s) for s in self.samples])
        ndx = dist.argsort()

        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        
        return max(votes)


def l2_dist(
    p1,
    p2
):
    return np.sqrt(np.sum((p1 - p2)**2))
