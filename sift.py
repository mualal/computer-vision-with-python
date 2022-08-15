import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def plot_sift_features(
    image_path: str,
    size: tuple,
) -> None:
    gray_img = np.array(Image.open(image_path).resize(size).convert('L'))
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray_img, None)

    def draw_circle(
        c: float,
        d: float,
    ) -> None:
        t = np.arange(0, 1.01, 0.01)
        x = d / 2 * np.cos(t) + c[0]
        y = d / 2 * np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)
    
    plt.figure()
    plt.imshow(gray_img)
    for point in kp:
        draw_circle((point.pt[0], point.pt[1]), point.size)
