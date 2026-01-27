import numpy as np

def convert_dB_to_linear(data: dict) -> dict:
    """
    Converts z-values in the data dictionary from dB to linear scale.

    Parameters:
    - data: A dictionary with keys 'x', 'y', and 'z' where 'z' contains values in dB.

    Returns:
    - A new dictionary with the same 'x' and 'y', but 'z' converted to linear scale.
    """
    linear_data = data.copy()
    linear_data['z'] = 10 ** (data['z'] / 10)
    return linear_data

def convert_linear_to_dB(data: dict) -> dict:
    """
    Converts z-values in the data dictionary from linear scale to dB.

    Parameters:
    - data: A dictionary with keys 'x', 'y', and 'z' where 'z' contains values in linear scale.

    Returns:
    - A new dictionary with the same 'x' and 'y', but 'z' converted to dB.
    """
    dB_data = data.copy()
    dB_data['z'] = 10 * np.log10(data['z'])
    return dB_data