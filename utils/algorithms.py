import numpy as np
import scipy as sp

import config_physics

def approximation_by_phys_model(x: np.ndarray, y: np.ndarray, model_func, initial_params: list, approximation_params: dict) -> list:
    """
    Fits the provided physical model to the data using non-linear least squares optimization.

    Parameters:
    x (np.ndarray): Input array of x values.
    y (np.ndarray): Input array of y values.
    model_func (callable): The physical model function to fit. It should take x and model parameters as arguments.
    initial_params (list): Initial guess for the model parameters.
    approximation_params (dict): Additional parameters for the approximation process, such as maxfev.

    Returns:
    list: Optimized model parameters.
    """
    p0 = initial_params
    popt, _ = sp.optimize.curve_fit(model_func, x, y, p0=p0, **approximation_params)
    return popt.tolist()


def find_peak(freqs, s_values, expected_freq, expected_width, expected_prominence, 
                       search_window=None, peak_type=None):
    """
    Гибкий поиск пика по примерным параметрам.
    
    Parameters:
    -----------
    freqs : array
        Массив частот (ГГц)
    s_values : array
        Массив значений S-параметра (должен быть по модулю)
    expected_freq : float
        Ожидаемая частота пика (ГГц)
    expected_width : float
        Примерная ширина пика (ГГц)
    expected_prominence : float
        Примерная высота/глубина пика (в единицах S-параметра)
    search_window : float, optional
        Ширина окна поиска вокруг expected_freq (ГГц). 
        По умолчанию = expected_width * 3
    peak_type : str, optional
        'maximum' или 'minimum'. Если None, берется из config_physics.PEAK_TYPE
    
    Returns:
    --------
    dict : {
        'freq': float - частота найденного пика,
        'magnitude': float - амплитуда S-параметра в пике,
        'prominence': float - prominence пика,
        'width': float - ширина пика (ГГц),
        'method': str - метод нахождения ('find_peaks' или 'extremum')
    }
    """
    if peak_type is None:
        peak_type = config_physics.PEAK_TYPE
    
    if search_window is None:
        search_window = expected_width * 3
    
    # Определяем окрестность для поиска
    freq_range = (expected_freq - search_window/2, expected_freq + search_window/2)
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    
    if len(freq_indices) == 0:
        raise ValueError(f"No data points in search window [{freq_range[0]:.4f}, {freq_range[1]:.4f}] GHz")
    
    # Извлекаем данные в окрестности
    local_freqs = freqs[freq_indices]
    local_s_values = s_values[freq_indices]

    if peak_type == 'minimum':
        local_s_values = -local_s_values  # Инвертируем для поиска минимумов
    
    # Вычисляем шаг по частоте
    freq_step = local_freqs[1] - local_freqs[0] if len(local_freqs) > 1 else 0.001
    
    # Преобразуем параметры в количество точек (с защитой от нуля)
    width_points = max(1, int(expected_width / freq_step / 2))  # Делим на 2 для мягкости
    
    # Пытаемся найти пик с помощью find_peaks с мягкими критериями
    found_peaks = sp.signal.find_peaks(
        local_s_values,
        prominence=expected_prominence * 0.5,  # Снижаем требование
        width=width_points
    )
    
    # Если find_peaks нашел пики
    if len(found_peaks[0]) > 0:
        # Выбираем самый яркий пик (с максимальной prominence)
        if len(found_peaks[0]) == 1:
            peak_idx = found_peaks[0][0]
            idx_in_result = 0
        else:
            # Находим индекс пика с максимальной prominence
            idx_in_result = np.argmax(found_peaks[1]['prominences'])
            peak_idx = found_peaks[0][idx_in_result]
        
        peak_freq = local_freqs[peak_idx]
        peak_magnitude = local_s_values[peak_idx] if peak_type == 'maximum' else -local_s_values[peak_idx]
        peak_prominence = found_peaks[1]['prominences'][idx_in_result]
        peak_width = found_peaks[1]['widths'][idx_in_result] * freq_step
        
        return {
            'freq': peak_freq,
            'magnitude': peak_magnitude,
            'prominence': peak_prominence,
            'width': peak_width,
            'method': 'find_peaks'
        }
    
    # Если find_peaks не нашел - ищем экстремум в окрестности
    # local_s_values уже инвертированы для минимумов, поэтому всегда ищем максимум
    peak_idx = np.argmax(local_s_values)
    
    peak_freq = local_freqs[peak_idx]
    # Возвращаем правильное значение magnitude с учётом инверсии
    peak_magnitude = local_s_values[peak_idx] if peak_type == 'maximum' else -local_s_values[peak_idx]
    
    # Оцениваем prominence вручную (разница между пиком и средним значением боковых частей)
    left_baseline = np.median(local_s_values[:max(1, peak_idx)]) if peak_idx > 0 else local_s_values[peak_idx]
    right_baseline = np.median(local_s_values[min(len(local_s_values)-1, peak_idx+1):]) if peak_idx < len(local_s_values)-1 else local_s_values[peak_idx]
    baseline = (left_baseline + right_baseline) / 2
    peak_prominence = abs(local_s_values[peak_idx] - baseline)
    
    # Используем исходную оценку ширины
    peak_width = expected_width
    
    return {
        'freq': peak_freq,
        'magnitude': peak_magnitude,
        'prominence': peak_prominence,
        'width': peak_width,
        'method': 'extremum'
    }


def make_cut(data: dict, cut_value: float, axis: str = 'x') -> dict:
    """
    Делает срез данных по заданному значению вдоль указанной оси.

    Parameters:
    data (dict): Словарь с данными, содержащий 'x', 'y', и 'z' массивы.
    cut_value (float): Значение, по которому делается срез.
    axis (str): Ось, вдоль которой делается срез ('x' или 'y').

    Returns:
    dict: Новый словарь с данными после среза.
    """
    x = data['x']
    y = data['y']
    z = data['z']

    if axis == 'x':
        idx = (np.abs(x - cut_value)).argmin()
        new_x = y
        new_y = z[idx, :]
        return {'x': new_x, 'y': new_y}
    elif axis == 'y':
        idx = (np.abs(y - cut_value)).argmin()
        new_x = x
        new_y = z[:, idx]
        return {'x': new_x, 'y': new_y}
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    

def estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato):
    
    con = np.abs(res_magnitude - plato)

    kappa = con * cavity_width / 2
    beta = cavity_width / 2 * (1 - con)

    return {'kappa': kappa,
            'beta': beta,
            'resonance_freq': resonance_freq,
            'plato': plato,
            'res_magnitude': res_magnitude}


def allighn_arrays(field1: np.ndarray, arr1: np.ndarray, field2: np.ndarray, arr2: np.ndarray):
    min_field = max(np.min(field1), np.min(field2))
    max_field = min(np.max(field1), np.max(field2))

    field_indexes1 = np.where((field1 >= min_field) & (field1 <= max_field))[0]
    field_indexes2 = np.where((field2 >= min_field) & (field2 <= max_field))[0]

    aligned_field = field1[field_indexes1]
    aligned_arr1 = arr1[field_indexes1]
    aligned_arr2 = arr2[field_indexes2]
    return aligned_field, aligned_arr1, aligned_arr2
    
  