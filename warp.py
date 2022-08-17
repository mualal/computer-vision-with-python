import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import homography


def image_to_image(
    im1: np.ndarray,
    im2: np.ndarray,
    to_points: np.ndarray,
) -> np.ndarray:
    m, n = im1.shape[:2]
    from_points = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])
    h = homography.haffine_from_points(to_points, from_points)
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
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    tri = Delaunay(np.c_[x, y]).simplices
    return tri


def pw_affine(
    from_im: np.ndarray,
    to_im: np.ndarray,
    from_points: np.ndarray,
    to_points: np.ndarray,
    tri: np.ndarray
) -> np.ndarray:
    im = to_im.copy()
    is_color = len(from_im.shape) == 3
    im_t = np.zeros(im.shape, 'uint8')

    for t in tri:
        h = homography.haffine_from_points(to_points[:, t], from_points[:, t])
        if is_color:
            for col in range(from_im.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(
                    from_im[:, :, col],
                    h[:2, :2],
                    (h[0, 2], h[1, 2]),
                    im.shape[:2]
                )
        else:
            im_t = ndimage.affine_transform(
                from_im,
                h[:2, :2],
                (h[0, 2], h[1, 2]),
                im.shape[:2]
            )
        alpha = alpha_for_triangle(to_points[:, t], im.shape[0], im.shape[1])
        im[alpha > 0] = im_t[alpha > 0]
    
    return im


def plot_mesh(
    x: np.ndarray,
    y: np.ndarray,
    tri: np.ndarray,
):
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]
        plt.plot(x[t_ext], y[t_ext], 'blue', linewidth=0.1)

