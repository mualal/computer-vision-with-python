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
):
    pil_im = Image.fromarray(np.uint8(arr))
    return np.array(pil_im.resize(target_size))


if __name__ == '__main__':
    images_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'images_input'
    )
    print(get_imlist(images_path))
    