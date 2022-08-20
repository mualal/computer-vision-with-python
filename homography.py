import numpy as np
import ransac


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
    from_points: np.ndarray,
    to_points: np.ndarray,
) -> np.ndarray:
    assert from_points.shape == to_points.shape

    # обеспечивает хорошую обусловленность для первого изображения
    fp, c1 = better_cond(from_points)
    # обеспечивает хорошую обусловленность для второго изображения
    tp, c2 = better_cond(to_points)

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
    from_points: np.ndarray,
    to_points: np.ndarray,
) -> np.ndarray:
    assert from_points.shape == to_points.shape

    # для первого изображения
    fp, c1 = better_cond(from_points)
    # для второго изображения
    tp, c2 = better_cond(to_points)

    a = np.concatenate((fp[:2], tp[:2]), axis=0)
    _, _, v = np.linalg.svd(a.T)

    tmp = v[:2].T
    b = tmp[:2]
    c = tmp[2:4]
    tmp2 = np.concatenate((np.dot(c, np.linalg.pinv(b)), np.zeros((2, 1))), axis=1)
    h = np.vstack((tmp2, [0, 0, 1]))
    h = np.dot(np.linalg.inv(c2), np.dot(h, c1))

    return h / h[2, 2]


class RansacModel:

    def __init__(
        self,
        debug=False
    ):
        self.debug = debug
    
    def fit(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        data = data.T
        from_points = data[:3, :4]
        to_points = data[3:, :4]
        return h_from_points(from_points, to_points)

    def get_error(
        self,
        data: np.ndarray,
        h: np.ndarray
    ) -> np.ndarray:
        data = data.T
        from_points = data[:3]
        to_points = data[3:]

        from_points_transformed = np.dot(h, from_points)
        for i in range(3):
            from_points_transformed[i] /= from_points_transformed[2]
        
        return np.sqrt(np.sum((to_points - from_points_transformed)**2, axis=0))


def h_from_ransac(
    from_points: np.ndarray,
    to_points: np.ndarray,
    model: RansacModel,
    max_iter=1000,
    match_threshold=10
) -> tuple:
    data = np.vstack((from_points, to_points))
    h, ransac_data = ransac.ransac(
        data=data.T,
        model=model,
        n=4,
        k=max_iter,
        t=match_threshold,
        d=10,
        return_all=True
    )
    return h, ransac_data['inliers']
