import pybaselines
import matplotlib.pyplot as plt
import numpy as np

def baseline_correction(waves, intensities, polyorder=2, numstd=1, plot=False):
    """
    Applies baseline correction to a single spectrum of intensity data.

    Parameters:
      - waves: array for the wavenumber values.
      - intensities: array of intensity values corresponding to each wavenumber.
      - plot: boolean indicating whether to plot the results.

    Returns:
      - intensities_corrected: array of baseline-corrected intensity values.
    """
    # Perform baseline correction
    baseline = pybaselines.polynomial.imodpoly(intensities, x_data=waves, poly_order = polyorder, num_std = numstd)
    baseline = np.array(baseline[0])  # Ensure baseline is a numpy array
    intensities_corrected = np.array(intensities) - baseline

    # Plot results if required
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(waves, intensities, label='Original Intensities', color='blue', linewidth=1)
        plt.plot(waves, baseline, label='Baseline', color='red', linestyle='--', linewidth=1)
        plt.plot(waves, intensities_corrected, label='Baseline Corrected Intensities', color='green', linewidth=1)
        plt.xlabel('Wavenumbers')
        plt.ylabel('Intensities')
        plt.title('Baseline Correction')
        plt.legend()
        plt.grid(True)
        plt.show()

    return intensities_corrected, baseline