import numpy as np

def filter_data(data, ranges):
    """
    Filters the input data based on the provided ranges.

    Parameters:
    - data: A dictionary with keys 'x', 'y', and 'z' where 'x' and 'y' are 1D arrays
            and 'z' is a 2D array corresponding to the grid defined by 'x' and 'y'.
    - ranges: A dictionary with optional keys 'x_range' and 'y_range', each containing
              a tuple (min, max) to define the filtering range.

    Returns:
    - A dictionary with filtered 'x', 'y', and 'z'.
    """
    x = np.array(data.get('x', []))
    y = np.array(data.get('y', []))
    z = np.array(data.get('z', []))

    if 'x_range' in ranges:
        x_min, x_max = ranges['x_range']
        x_mask = (x >= x_min) & (x <= x_max)
        x = x[x_mask]
        z = z[x_mask, :]

    if 'y_range' in ranges:
        y_min, y_max = ranges['y_range']
        y_mask = (y >= y_min) & (y <= y_max)
        y = y[y_mask]
        z = z[:, y_mask]

    filtered_data = data.copy()
    filtered_data.update({'x': x, 'y': y, 'z': z})
    return filtered_data