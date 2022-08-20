import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def find_sift_points_and_descriptors(
    img: np.ndarray
) -> tuple:
    sift = cv.SIFT_create()
    keypoints, des = sift.detectAndCompute(img, None)
    return np.array(keypoints), des


def plot_sift_points(
    image_path: str,
    size: tuple,
) -> None:
    gray_img = np.array(Image.open(image_path).resize(size).convert('L'))
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    keypoints, _ = find_sift_points_and_descriptors(gray_img)

    def draw_circle(
        c: float,
        d: float,
    ) -> None:
        t = np.arange(0, 1.01, 0.01)
        x = d / 2 * np.cos(t) + c[0]
        y = d / 2 * np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)
    
    plt.imshow(gray_img)
    for point in keypoints:
        draw_circle((point.pt[0], point.pt[1]), point.size)


def match(
    desc1: np.ndarray,
    desc2: np.ndarray,
) -> np.ndarray:
    desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d / np.linalg.norm(d) for d in desc2])
    dist_ratio = 0.6
    desc1_size = desc1.shape
    matchscores = np.zeros(desc1_size[0], 'int')
    desc2_t = desc2.T
    for i in range(desc1_size[0]):
        dot_prods = np.dot(desc1[i, :], desc2_t)
        dot_prods = 0.9999 * dot_prods
        indx = np.argsort(np.arccos(dot_prods))
        if np.arccos(dot_prods)[indx[0]] < dist_ratio * np.arccos(dot_prods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores


def match_twosided(
    desc1: np.ndarray,
    desc2: np.ndarray,
) -> np.ndarray:
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)
    ndx_12 = matches_12.nonzero()[0]

    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    return matches_12


def concatenate_images(
    im1: np.ndarray,
    im2: np.ndarray,
) -> np.ndarray:
    """
    конкатенация двух изображений в одно новое изображение
    @param im1: первое изображение в виде numpy-массива
    @param im2: второе изображение в виде numpy-массива
    @return: объединённое изображение
    """
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    # если количество пиксельных строк в изображениях не совпадают, то дополняем нулями
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    
    return np.concatenate((im1, im2), axis=1)


def plot_matches(
    im1: np.ndarray,
    im2: np.ndarray,
    locs1: list,
    locs2: list,
    matchscores: list,
    show_below=True,
) -> None:
    """
    вывод изображения, на котором соответственные точки соединены друг с другом
    @param im1: первое изображение в виде numpy-массива
    @param im2: второе изображение в виде numpy-массива
    @param locs1: координаты особых точек на первом изображении
    @param locs2: координаты особых точек на втором изображении
    @param matchscores: результат функции match_descriptors
    @param show_below: необходимо ли показать исходные изображения под картиной соответствия
    """
    plt.figure()
    plt.gray()
    im3 = concatenate_images(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))
    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plt.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
    plt.axis('off')
    plt.show()
