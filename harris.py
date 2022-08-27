import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt


def compute_harris_response(
    im: np.ndarray,
    sigma=3,
) -> np.ndarray:
    """
    вычисляет функцию отклика детектора углов Харриса для каждого пикселя
    полутонового изображения
    @param im: изображение в форме numpy-массива
    @param sigma: масштаб фильтров
    @return: массив со значениями функции отклика
    """
    # производные
    im_x = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), im_x)
    im_y = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), im_y)

    # элементы матрицы Харриса
    W_xx = filters.gaussian_filter(im_x * im_x, sigma)
    W_xy = filters.gaussian_filter(im_x * im_y, sigma)
    W_yy = filters.gaussian_filter(im_y * im_y, sigma)

    # определитель и след матрицы
    W_det = W_xx * W_yy - W_xy**2
    W_tr = W_xx + W_yy

    return W_det / W_tr


def get_harris_points(
    harris_im: np.ndarray,
    min_dist=10,
    threshold=0.1,
) -> list:
    """
    находит координаты углов
    @param harris_im: изображение, построенное по функции отклика Харриса (в виде numpy-массива)
    @param min_dist: минимальное количество пикселей между соседними углами
    @param threshold: отсечка
    @return: список с координатами найденных углов
    """
    # точки-координаты, для которых функция отклика больше порога
    corner_threshold = harris_im.max() * threshold
    harris_im_detect = harris_im > corner_threshold
    # координаты кандидатов
    coords = np.array(harris_im_detect.nonzero()).T
    # значения кандидатов
    candidate_values = [harris_im[c[0], c[1]] for c in coords]
    # сортировка кандидатов
    index = np.argsort(candidate_values)
    # запись данных о точках-кандидатах в массив
    allowed_locations = np.zeros(harris_im.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    # выбор углов с учётом минимального количества пикселей между ними
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
                (coords[i,0]-min_dist):(coords[i,0]+min_dist),
                (coords[i,1]-min_dist):(coords[i,1]+min_dist)
            ] = 0
    
    return filtered_coords


def plot_harris_points(
    image: np.ndarray,
    filtered_coords: list,
) -> None:
    """
    вывод изображения с углами, координаты которых заданы
    @param image: исходное изображение в виде numpy-массива
    @param filtered_coords: координаты углов
    """
    plt.figure()
    plt.gray()
    plt.imshow(image)
    # добавление найденных углов
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()


def get_descriptors(
    image: np.ndarray,
    filtered_coords: list,
    wid=5,
) -> list:
    """
    для каждой точки возвращает значения пикселей в окрестности этой точки шириной 2 * wid + 1
    (предполагается, что выбирались точки с min_distance > wid)
    @param image: исходное изображение в виде numpy-массива
    @param filtered_coords: координаты углов
    @param wid: полудлина окна (блока изображения)
    @return: список дескрипторов
    """
    desc = []
    for coords in filtered_coords:
        patch = image[
            coords[0]-wid:coords[0]+wid+1,
            coords[1]-wid:coords[1]+wid+1
        ].flatten()
        desc.append(patch)
    return desc


def match_descriptors(
    desc1: list,
    desc2: list,
    threshold=0.5,
) -> list:
    """
    для каждого дескриптора угловой точки на первом изображении находит соответствующую
    ему точку на втором изображении, применяя нормированную взаимную корреляцию
    @param desc1: список дескрипторов углов первого изображения
    @param desc2: список дескрипторов углов второго изображения
    @param threshold: отсечка
    @return: список из значений соответствия
    """
    n = len(desc1[0])
    d = - np.ones((len(desc1), len(desc2)))
    for i, _ in enumerate(desc1):
        for j, _ in enumerate(desc2):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = np.sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    ndx = np.argsort(-d)
    match_scores = ndx[:, 0]
    return match_scores


def match_descriptors_twosided(
    desc1: list,
    desc2: list,
    threshold=0.5,
) -> list:
    matches_12 = match_descriptors(desc1, desc2, threshold)
    matches_21 = match_descriptors(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # исключение несимметричных соответствий
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
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
