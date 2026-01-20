def choose_one_point(event, params):
    if event.inaxes is None:
        return params, False  # Click was outside axes, do nothing

    x, y = event.xdata, event.ydata
    output = {'x': x, 'y': y, 'marker': 'o'}
    params.update({'select_one_point': output})
    return params, True  # Reset mode after selection