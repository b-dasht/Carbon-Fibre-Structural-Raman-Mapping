import numpy as np

def extract_spectra(data, x_value, y_value, col1_name, col2_name):
    """
    Extracts the wave and intensity data for a specific X and Y combination from the data
    and normalises the intensity data to the range [0, 100].

    Parameters:
    - data: pd.DataFrame, the DataFrame containing the Raman spectroscopy data.
    - x_value: float or int, the X value to filter on.
    - y_value: float or int, the Y value to filter on.

    Returns:
    - A tuple (waves, intensities), where waves and intensities are numpy arrays of the wave and normalized intensity data.
    """
    # Filter the data for the specific (X, Y) combination
    filtered_data = data[(data['X'] == x_value) & (data['Y'] == y_value)]

    # Extract the wave and intensity columns
    waves = filtered_data['Wave'].values
    raw_intensities = filtered_data[str(col1_name)].values
    denoised_intensities = filtered_data[str(col2_name)].values

    return np.array(waves, dtype=float), np.array(raw_intensities, dtype=float), np.array(denoised_intensities, dtype=float)
