import numpy as np
from scipy import linalg


class Camera:
    def __init__(
        self,
        p: np.ndarray,
    ) -> None:
        self.p = p
        self.k = None  # калибровочная матрица
        self.r = None  # поворот
        self.t = None  # параллельный перенос
        self.c = None  # центр камеры
    
    def project(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x = np.dot(self.p, x)
        for i in range(3):
            x[i] /= x[2]
        return x
    
    def factor(
        self,
    ) -> tuple:
        k, r = linalg.rq(self.p[:, :3])
        t = np.diag(np.sign(np.diag(k)))
        if linalg.det(t) < 0:
            t[1, 1] *= -1
        self.k = np.dot(k, t)
        self.r = np.dot(t, r)  # обратная к t матрица совпадает с ней самой
        self.t = np.dot(linalg.inv(self.k), self.p[:, :3])
        return self.k, self.r, self.t
    
    def center(
        self,
    ) -> np.ndarray:
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c = - np.dot(self.r.T, self.t)
            return self.c


def rotation_matrix(
    a: np.ndarray,
) -> np.ndarray:
    """
    создаёт матрицу поворота вокруг оси вектора a в трёхмерном пространстве
    """
    r = np.eye(4)
    r[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return r
