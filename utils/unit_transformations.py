import numpy as np

def convert_dB_to_linear(x):
    """
    Converts z-values in the data dictionary from dB to linear scale.

    Parameters:
    - data: A dictionary with keys 'x', 'y', and 'z' where 'z' contains values in dB.

    Returns:
    - A new dictionary with the same 'x' and 'y', but 'z' converted to linear scale.
    """

    linear = 10 ** (x / 10)
    return linear

def convert_linear_to_dB(x):
    """
    Converts z-values in the data dictionary from linear scale to dB.

    Parameters:
    - data: A dictionary with keys 'x', 'y', and 'z' where 'z' contains values in linear scale.

    Returns:
    - A new dictionary with the same 'x' and 'y', but 'z' converted to dB.
    """
    # Handle invalid values (negative or zero)
    x = np.asarray(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        dB = 10 * np.log10(np.abs(x))
    # Replace inf/-inf with very small/large values
    dB = np.where(np.isfinite(dB), dB, -300)
    return dB