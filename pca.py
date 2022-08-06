import numpy as np


def pca(
    flattened_arrays: np.ndarray,
) -> tuple:

    # количество изображений и размерность изображения
    flattened_arrays_count, dim = flattened_arrays.shape

    # центрирование изображений (сглаженных массивов) относительно некого среднего изображения
    mean_flattened_array = flattened_arrays.mean(axis=0)
    flattened_arrays = flattened_arrays - mean_flattened_array

    if dim > flattened_arrays_count:
        # PCA с компактным трюком
        M = np.dot(flattened_arrays, flattened_arrays.T)  # ковариационная матрица
        eigenvalues, eigenvectors = np.linalg.eigh(M)  # собственные значения и векторы
        temp = np.dot(flattened_arrays.T, eigenvectors).T  # компактный трюк
        vectors = temp[::-1]
        values = np.sqrt(eigenvalues)[::-1]

        for i in range(vectors.shape[1]):
            vectors[:, i] /= values
    else:
        # PCA с использованием сингулярного разложения
        _, values, vectors = np.linalg.svd(flattened_arrays)
        vectors = vectors[:flattened_arrays_count]
    
    return vectors, values, mean_flattened_array
