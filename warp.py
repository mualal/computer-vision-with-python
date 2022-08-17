import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
import homography


def image_to_image(
    im1: np.ndarray,
    im2: np.ndarray,
    tp: np.ndarray,
) -> np.ndarray:
    m, n = im1.shape[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])
    h = homography.haffine_from_points(tp, fp)
    im1_transformed = ndimage.affine_transform(im1, h[:2, :2], (h[0, 2], h[1, 2]), im2.shape[:2])
    alpha = (im1_transformed > 0)

    return (1 - alpha) * im2 + alpha * im1_transformed


def alpha_for_triangle(
    points: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    alpha = np.zeros((m, n))
    for i in range(int(min(points[0])), int(max(points[0]))):
        for j in range(int(min(points[1])), int(max(points[1]))):
            x = np.linalg.solve(points, [i, j, 1])
            if min(x) > 0:
                alpha[i, j] = 1
    return alpha


def triangulate_points(
    x,
    y
):
    tri = Delaunay(np.c_[x, y]).simplices
    return tri
