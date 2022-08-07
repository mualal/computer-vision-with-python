import numpy as np


def denoise(
    im: np.ndarray,
    u_init: np.ndarray,
    tolerance=0.1,
    tau=0.125,
    tv_weight=100,
) -> tuple:
    """
    Очистка от шумов; используются модель Рудина-Ошера-Фатеми (ROF) и алгоритм Шамболя.
    Другими словами, решатель ROF на базе алгоритма Шамболя (Chambolle)
    @param im: зашумлённое полутоновое изображение
    @param u_init: начальное значение U
    @param tolerance: допуск в условии остановки
    @param tau: величина шага
    @param tv_weight: вес слагаемого, регуляризирующего TV
    @return: очищенное от шумов изображение и остаточная текстура
    """
    m, n = im.shape  # размер зашумлённого изображения
    u = u_init  # инициализация
    p_x = im  # компонента x двойственной задачи
    p_y = im  # компонента y двойственной задачи
    error = 1

    while error > tolerance:
        u_old = u
        # градиент переменной прямой задачи
        grad_u_x = np.roll(u, -1, axis=1) - u  # компонента x градиента U
        grad_u_y = np.roll(u, -1, axis=0) - u  # компонента y градиента U
        # изменение переменной двойственной задачи
        p_x_new = p_x + (tau / tv_weight) * grad_u_x
        p_y_new = p_y + (tau / tv_weight) * grad_u_y
        norm_new = np.maximum(1, np.sqrt(p_x_new**2 + p_y_new**2))
        p_x = p_x_new / norm_new
        p_y = p_y_new / norm_new
        # изменение переменной прямой задачи
        r_x_p_x = np.roll(p_x, 1, axis=1)  # циклический сдвиг компоненты x вдоль оси x
        r_y_p_y = np.roll(p_y, 1, axis=0)  # циклический сдвиг компоненты y вдоль оси y
        div_p = (p_x - r_x_p_x) + (p_y - r_y_p_y)  # дивергенция двойственного поля
        u = im + tv_weight * div_p  # изменение переменной прямой задачи
        error = np.linalg.norm(u - u_old) / np.sqrt(n * m)  # пересчёт погрешности
    
    return u, im - u
