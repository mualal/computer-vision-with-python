import os
import numpy as np
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
    arr_hist, bins = np.histogram(arr.flatten(), bins_count, normed=True)
    cdf = arr_hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    new_arr = np.interp(arr.flatten(), bins[:-1], cdf)
    return new_arr.reshape(arr.shape), cdf


def image_averaging(
    images_paths: list[str]
) -> np.ndarray:
    average_image = np.array(Image.open(images_path[0]), 'f')
    for image_path in images_paths[1:]:
        try:
            average_image += np.array(Image.open(image_path))
        except:
            print(image_path + ' failed to open')
            average_image /= len(images_paths)
    return np.array(average_image, 'uint8')


if __name__ == '__main__':
    images_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'images_input'
    )
    print(get_imlist(images_path))
    