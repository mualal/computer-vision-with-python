import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_imlist(
    path: str
) -> list:
    """
    получить список имён всех jpg-файлов в каталоге
    @param path: путь к каталогу
    @return: список имён всех jpg-файлов в каталоге
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') and not f.startswith('.')]


def arr_resize(
    arr: np.array,
    target_size: tuple,
) -> np.array:
    """
    изменить размер массива с помощью PIL
    @param arr: numpy-массив
    @param target_size: целевой размер массива
    @return: numpy-массив с размером, приведённым к целевому
    """
    pil_im = Image.fromarray(np.uint8(arr))
    return np.array(pil_im.resize(target_size))


def hist_equalization(
    arr: np.array,
    bins_count=256,
) -> tuple:
    """
    выравнивание гистограммы полутонового изображения
    @param arr: массив изображения
    @param bins_count: количество интервалов у гистограммы
    @return: изображение с выровненной гистограммой и значения функции распределения
    """
    # получить гистограмму изображения
    arr_hist, bins = np.histogram(arr.flatten(), bins_count)
    # функция распределения значений пикселей в изображении
    cdf = arr_hist.cumsum()
    # (нормированная так, чтобы привести значения к требуемому диапазону)
    cdf = 255 * cdf / cdf[-1]

    # использовать линейную интерполяцию функции распределения (cdf)
    # для нахождения значений новых пикселей
    new_arr = np.interp(arr.flatten(), bins[:-1], cdf)
    return new_arr.reshape(arr.shape), cdf


def image_averaging(
    images_paths: list
) -> np.ndarray:
    """
    вычислить среднее списка изображений
    @param images_paths: список с путями к изображениям
    @return: среднее изображение
    """
    # открыть первое изображение и преобразовать
    # в массив типа float
    average_image = np.array(Image.open(images_paths[0]), 'f')
    success_count = len(images_paths)
    for image_path in images_paths[1:]:
        try:
            average_image += np.array(Image.open(image_path))
        except:
            print(image_path + ' failed to open')
            success_count -= 1
    average_image /= success_count
    return np.array(average_image, 'uint8')


def plot_2d_boundary(
    plot_range: tuple,
    points: list,
    decisionfcn,
    labels: np.ndarray,
    values=[0]
) -> None:
    """
    визуализировать классификацию всех тестовых точек и показать,
    насколько хорошо классификатор разделяет 2 класса
    @param plot_range: диапазон (xmin, xmax, ymin, ymax)
    @param points: список точек
    @param decisionfcn: функция, принимающая решение
    @param labels: массив меток
    @param values: список изолиний для отображения
    @return: None
    """
    clist = ['b', 'r', 'g', 'k', 'm', 'y']
    x = np.arange(plot_range[0], plot_range[1], .1)
    y = np.arange(plot_range[2], plot_range[3], .1)
    xx, yy = np.meshgrid(x, y)
    xxx, yyy = xx.flatten(), yy.flatten()
    zz = np.array(decisionfcn(xxx, yyy))
    zz = zz.reshape(xx.shape)
    plt.contour(xx, yy, zz, values)
    for i, _ in enumerate(points):
        d = decisionfcn(points[i][:, 0], points[i][:, 1])
        correct_ndx = labels[i]==d
        incorrect_ndx = labels[i]!=d
        plt.plot(
            points[i][correct_ndx, 0],
            points[i][correct_ndx, 1],
            '*', 
            color=clist[i]
        )
        plt.plot(
            points[i][incorrect_ndx, 0],
            points[i][incorrect_ndx, 1],
            'o', 
            color=clist[i]
        )
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    images_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'images_input'
    )
    print(get_imlist(images_path))
    