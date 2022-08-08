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
