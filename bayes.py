import numpy as np


class BayesClassifier:
    """
    Наивный байесовский классификатор с использованием модели нормального распределения вероятностей
    """
    def __init__(
        self
    ) -> None:
        self.labels = []  # метки классов
        self.mean = []  # средние классов
        self.var = []  # дисперсии классов
        self.n = 0  # число классов
    
    def train(
        self,
        data: list,
        labels=None
    ) -> None:
        """
        обучить на данных
        @param data: список массивов для обучения классификатора
        @param labels: названия (обозначения) классов
        @return: None
        """
        if labels is None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)
        for c in data:
            self.mean.append(np.mean(c, axis=0))
            self.var.append(np.var(c, axis=0))
    
    def classify(
        self,
        points: np.ndarray
    ) -> tuple:
        """
        вычислить вероятности принадлежности точек points к каждому классу
        @param points: 
        @return: метки самого вероятного класса для каждой точки и вероятности принадлежности
        к каждому классу
        """
        est_prob = np.array([multi_gauss(m, v, points) for m, v in zip(self.mean, self.var)])
        ndx = est_prob.argmax(axis=0)
        est_labels = np.array([self.labels[n] for n in ndx])

        return est_labels, est_prob


def multi_gauss(
    m: float,
    v: float,
    x: np.ndarray
) -> float:
    """
    вычислить d-мерное нормальное распределение со средним m и дисперсией v в точках (строках) x
    @param m: среднее
    @param v: дисперсия
    @param x:
    @return: 
    """
    if len(x.shape) == 1:
        _, d = 1, x.shape[0]
    else:
        _, d = x.shape
    
    s = np.diag(1 / v)
    x = x - m
    y = np.exp(-0.5 * np.diag(np.dot(x, np.dot(s, x.T))))

    return y * (2 * np.pi)**(-d / 2.0) / (np.sqrt(np.prod(v)) + 1e-6)
