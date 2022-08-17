import numpy as np


def normalize(
    points: np.ndarray,
) -> np.ndarray:
    for row in points:
        row /= points[-1]
    return points


def make_homog(
    points: np.ndarray
) -> np.ndarray:
    return np.vstack((points, np.ones((1, points.shape[1]))))


def better_cond(
    matrix: np.ndarray
) -> tuple:
    m = np.mean(matrix[:2], axis=1)
    max_std = max(np.std(matrix[:2], axis=1)) + 1e-9
    c = np.diag([1 / max_std, 1 / max_std, 1])
    c[0][2] = - m[0] / max_std
    c[1][2] = - m[1] / max_std
    return np.dot(c, matrix), c


def h_from_points(
    fp: np.ndarray,
    tp: np.ndarray,
) -> np.ndarray:
    assert fp.shape == tp.shape

    # обеспечивает хорошую обусловленность для первого изображения
    fp, c1 = better_cond(fp)
    # обеспечивает хорошую обусловленность для второго изображения
    tp, c2 = better_cond(tp)

    # создание матрицы (по две строки для каждой пары соответственных точек)
    nbr_correspondences = fp.shape[1]
    a = np.zeros((2 * nbr_correspondences, 9))
    # наполнение матрицы
    for i in range(nbr_correspondences):
        a[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, fp[0][i]*tp[0][i], fp[1][i]*tp[0][i], tp[0][i]]
        a[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, fp[0][i]*tp[1][i], fp[1][i]*tp[1][i], tp[1][i]]
    # находим значение H, минимизирующее среднеквадратичную ошибку, применив метод сингулярного разложения (SVD)
    _, _, v = np.linalg.svd(a)
    h = v[8].reshape((3, 3))

    # обратное преобразование, компенсирующее обуславливание
    h = np.dot(np.linalg.inv(c2), np.dot(h, c1))

    return h / h[2, 2]


def haffine_from_points(
    fp: np.ndarray,
    tp: np.ndarray,
) -> np.ndarray:
    assert fp.shape == tp.shape

    # для первого изображения
    fp, c1 = better_cond(fp)
    # для второго изображения
    tp, c2 = better_cond(tp)

    a = np.concatenate((fp[:2], tp[:2]), axis=0)
    _, _, v = np.linalg.svd(a.T)

    tmp = v[:2].T
    b = tmp[:2]
    c = tmp[2:4]
    tmp2 = np.concatenate((np.dot(c, np.linalg.pinv(b)), np.zeros((2, 1))), axis=1)
    h = np.vstack((tmp2, [0, 0, 1]))
    h = np.dot(np.linalg.inv(c2), np.dot(h, c1))

    return h / h[2, 2]
