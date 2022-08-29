from xml.dom import minidom
from matplotlib.image import imsave
from scipy import linalg, ndimage
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def read_points_from_xml(
    xml_filename: str
) -> np.ndarray:
    """
    считать контрольные точки для совмещения лиц
    @param xml_filename: имя xml-файла
    @return: словарь с опорными точками: ключ словаря совпадает с именем файла,
    а значениями являются массивы с опорными точками (координаты левого глаза,
    координаты правого глаза и координаты рта в декартовой СК)
    """
    xml_doc = minidom.parse(xml_filename)
    facelist = xml_doc.getElementsByTagName('face')
    faces = {}
    for xml_face in facelist:
        filename = xml_face.attributes['file'].value
        xf = int(xml_face.attributes['xf'].value)
        yf = int(xml_face.attributes['yf'].value)
        xs = int(xml_face.attributes['xs'].value)
        ys = int(xml_face.attributes['ys'].value)
        xm = int(xml_face.attributes['xm'].value)
        ym = int(xml_face.attributes['ym'].value)
        faces[filename] = np.array([xf, yf, xs, ys, xm, ym])
    return faces


def compute_rigit_transform(
    refpoints,
    points
):
    """
    вычислить угол поворота, коэффициент масштабирования и вектор параллельного
    переноса для совмещения пар опорных точек
    @param refpoints: базовые опорные точки
    @param points: опорные точки для совмещения
    @return: матрица поворота с масштабированием, координаты вектора параллельного переноса
    """
    a = np.array(
        [[points[0], -points[1], 1, 0],
         [points[1], points[0], 0, 1],
         [points[2], -points[3], 1, 0],
         [points[3], points[2], 0, 1],
         [points[4], -points[5], 1, 0],
         [points[5], points[4], 0, 1]]
    )

    y = np.array(
        [refpoints[0],
         refpoints[1],
         refpoints[2],
         refpoints[3],
         refpoints[4],
         refpoints[5]]
    )

    # метод наименьших квадратов минимизирует норму ||Ax - y||
    a, b, tx, ty = linalg.lstsq(a, y)[0]
    # матрица поворота с масштабированием
    r = np.array(
        [[a, -b],
         [b, a]]
    )

    return r, tx, ty


def rigit_alignment(
    faces,
    path,
    plot_flag=False
) -> None:
    refpoints = list(faces.values())[0]

    for face in faces:
        points = faces[face]

        r, tx, ty = compute_rigit_transform(refpoints, points)
        t = np.array(
            [[r[1][1], r[1][0]],
            [r[0][1], r[0][0]]]
        )

        im = np.array(Image.open(os.path.join(path, face)))
        im2 = np.zeros(im.shape, 'uint8')

        # деформировать каждый цветовой канал
        for i, _ in enumerate(im.shape):
            im2[:, :, i] = ndimage.affine_transform(
                im[:, :, i],
                linalg.inv(t),
                offset=[-ty, -tx]
            )
        
        if plot_flag:
            plt.imshow(im2)
            plt.show()
        
        h, w = im2.shape[:2]
        border = (w + h) // 20
        imsave(os.path.join(path, 'aligned', face), im2[border:h-border, border:w-border, :])
