import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_imlist(
    path: str
) -> list:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') and not f.startswith('.')]


def arr_resize(
    arr: np.array,
    target_size: tuple,
) -> np.array:
    pil_im = Image.fromarray(np.uint8(arr))
    return np.array(pil_im.resize(target_size))


def hist_equalization(
    arr: np.array,
    bins_count=256,
) -> tuple:
    arr_hist, bins = np.histogram(arr.flatten(), bins_count)
    cdf = arr_hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    new_arr = np.interp(arr.flatten(), bins[:-1], cdf)
    return new_arr.reshape(arr.shape), cdf


def image_averaging(
    images_paths: list
) -> np.ndarray:
    average_image = np.array(Image.open(images_paths[0]), 'f')
    for image_path in images_paths[1:]:
        try:
            average_image += np.array(Image.open(image_path))
        except:
            print(image_path + ' failed to open')
            average_image /= len(images_paths)
    return np.array(average_image, 'uint8')


def plot_2d_boundary(
    plot_range,
    points,
    decisionfcn,
    labels,
    values=[0]
) -> None:
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
    