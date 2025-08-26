import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fit_and_save_spectra(waves, intensities, x_val, y_val):
    """
    Fits the spectra, calculates I-D/I-G ratios, saves data to files, and returns relevant fitting details.

    Parameters:
    - waves: array-like, wavelengths of the Raman spectra.
    - intensities: array-like, intensity values corresponding to the wavelengths.
    - x_val: float, X coordinate value.
    - y_val: float, Y coordinate value.

    Returns:
    - y_fit_id: array, fitted values for I-D peak.
    - y_fit_ig: array, fitted values for I-G peak.
    - y_fit_id2: array, fitted values for I-D2 peak.
    - y_fit_ig2: array, fitted values for I-G2 peak.
    - y_fit_combined: array, combined fitted values.
    - r_squared: float, goodness of fit measure.
    - ratio_id_ig_intensity: float, intensity ratio of I-D to I-G.
    - ratio_id_ig_area: float, area ratio of I-D to I-G.
    """
    # Gaussian function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


    def fit_id_ig(x_data, y_data):
        # Initial guesses for I-D and I-G
        initial_guess_id_ig = [
            max(y_data) * 0.4, 1400, 50,  # I-D: amplitude, mean, stddev
            max(y_data) * 0.4, 1600, 30   # I-G: amplitude, mean, stddev
        ]

        # Define bounds for parameters
        lower_bounds = [
            0, 1350, 20,
            0, 1550, 20
        ]
        upper_bounds = [
            np.inf, 1450, 60,
            np.inf, 1650, 60
        ]

        # Perform curve fitting using the set bounds
        popt_id_ig, _ = curve_fit(
            lambda x, amp1, mean1, stddev1, amp2, mean2, stddev2:
            gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2),
            x_data, y_data, p0=initial_guess_id_ig, bounds=(lower_bounds, upper_bounds), maxfev=5000
        )

        # Extract the parameters for the fitted I-D and I-G
        amp1, mean1, stddev1, amp2, mean2, stddev2 = popt_id_ig

        # Calculate areas
        area_id = amp1 * stddev1 * np.sqrt(2 * np.pi)
        area_ig = amp2 * stddev2 * np.sqrt(2 * np.pi)

        return amp1, mean1, stddev1, amp2, mean2, stddev2, area_id, area_ig

    def fit_id2_ig2(x_data, y_data, amp1, mean1, stddev1, amp2, mean2, stddev2):
        # Initial guesses for I-D2 and I-G2
        initial_guess_id2_ig2 = [
            max(y_data) * 0.1, 1320, 50,  # I-D2: amplitude, mean, stddev
            max(y_data) * 0.1, 1500, 50   # I-G2: amplitude, mean, stddev
        ]

        # Define bounds for the second fitting (with restrictions based on the first fit)
        lower_bounds = [
            amp1 * 0.5, mean1 - 10, 20,  # Lower bounds for I-D
            amp2 * 0.5, mean2 - 20, 20,  # Lower bounds for I-G
            0, 1250, 30,            # Lower bounds for I-D2
            0, 1450, 30             # Lower bounds for I-G2
        ]
        upper_bounds = [
            amp1 * 1.4, mean1 + 10, 80,     # Upper bounds for I-D
            amp2 * 1.4, mean2 + 20, 80,     # Upper bounds for I-G
            max(y_data) * 0.25, 1380, 80,  # Upper bounds for I-D2
            max(y_data) * 0.25, 1580, 80   # Upper bounds for I-G2
        ]

        # Perform curve fitting of fourt components with the set bounds
        popt_all, _ = curve_fit(
            lambda x, amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4:
            gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2) +
            gaussian(x, amp3, mean3, stddev3) + gaussian(x, amp4, mean4, stddev4),
            x_data, y_data,
            p0=[amp1, mean1, stddev1, amp2, mean2, stddev2, *initial_guess_id2_ig2],
            bounds=(lower_bounds, upper_bounds), maxfev=20000
        )

        # Extract the best-fit parameters for all peaks
        amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4 = popt_all

        return amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4

    def calculate_r_squared(y_data, y_fit_combined, x_data):
        # Filter data between wavenumbers of 1200 and 1700 (to avoid edge issues)
        mask = (x_data >= 1200) & (x_data <= 1700)
        y_data_filtered = y_data[mask]
        y_fit_filtered = y_fit_combined[mask]

        # Residual sum of squares
        ss_res = np.sum((y_data_filtered - y_fit_filtered) ** 2)

        # Calculate total sum of squares
        ss_tot = np.sum((y_data_filtered - np.mean(y_data_filtered)) ** 2)

        # Calculate R² goodness of fit
        r_squared = 1 - (ss_res / ss_tot)

        return r_squared

    # Convert input arrays to dataframe
    df = pd.DataFrame({'Wave': waves, 'Intensity': intensities})

    # Crop data to intensity ranges between 1000-1800
    crop_mask = (df['Wave'] >= 1000) & (df['Wave'] <= 1800)
    x_data = df['Wave'][crop_mask].values
    y_data_corrected = df['Intensity'][crop_mask].values

    # # Normalize y_data to the range 0-100. Outdated
    # y_data_min = np.min(y_data_corrected)
    # y_data_max = np.max(y_data_corrected)
    # y_data_corrected = 100 * (y_data_corrected - y_data_min) / (y_data_max - y_data_min)

    # Initialize fitted component arrays with zeros
    y_fit_id = np.zeros_like(intensities)
    y_fit_ig = np.zeros_like(intensities)
    y_fit_id2 = np.zeros_like(intensities)
    y_fit_ig2 = np.zeros_like(intensities)
    y_fit_combined = np.zeros_like(intensities)

    # Perform the initial I-D and I-G fitting
    amp1, mean1, stddev1, amp2, mean2, stddev2, area_id, area_ig = fit_id_ig(x_data, y_data_corrected)

    # Extract the fitted parameters
    amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4 = fit_id2_ig2(x_data, y_data_corrected, amp1, mean1, stddev1, amp2, mean2, stddev2)

    # Create Gaussian plots for the individual components and combined fit
    y_fit_id = gaussian(df['Wave'].values, amp1, mean1, stddev1)
    y_fit_ig = gaussian(df['Wave'].values, amp2, mean2, stddev2)
    y_fit_id2 = gaussian(df['Wave'].values, amp3, mean3, stddev3)
    y_fit_ig2 = gaussian(df['Wave'].values, amp4, mean4, stddev4)
    y_fit_combined = y_fit_id + y_fit_ig + y_fit_id2 + y_fit_ig2

    # Calculate R² goodness of fit
    r_squared = calculate_r_squared(y_data_corrected, y_fit_combined[crop_mask], x_data)

    # Remove fits with R² values below 0.6 and calculate I-D/I-G intensity ratio
    if r_squared > 0.6:
        ratio_id_ig_intensity = amp1 / amp2 if amp1 != 0 and amp2 != 0 else 0
        area_id = amp1 * stddev1 * np.sqrt(2 * np.pi) if amp1 != 0 and stddev1 != 0 else 0 # Area of the fitted I-D peak
        area_ig = amp2 * stddev2 * np.sqrt(2 * np.pi) if amp2 != 0 and stddev2 != 0 else 0 # Area of the fitted I-G peak
    else:
        # Set the values to 0 if the R² is below threshold
        ratio_id_ig_intensity = 0
        area_id = 0
        area_ig = 0

    # Calculate I-D/I-G area ratio
    ratio_id_ig_area = area_id / area_ig if area_id != 0 and area_ig != 0 else 0

    return y_fit_id, y_fit_ig, y_fit_id2, y_fit_ig2, y_fit_combined, r_squared, ratio_id_ig_intensity, ratio_id_ig_area
