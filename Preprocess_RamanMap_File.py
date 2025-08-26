import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import os

def baseline_correction(waves, intensities, fixed_points, poly_order=2):
    """
    Applies baseline correction to the intensity data using a polynomial fit based on fixed points.
    """
    fixed_points = np.array(fixed_points)
    valid_points = (fixed_points >= min(waves)) & (fixed_points <= max(waves))
    fixed_points = fixed_points[valid_points]

    baseline_values = []
    num_points_to_average = 20

    for fp in fixed_points:
        idx = np.argmin(np.abs(waves - fp))
        start_idx = max(idx - num_points_to_average // 2, 0)
        end_idx = min(idx + num_points_to_average // 2 + 1, len(intensities))
        average_intensity = np.mean(intensities[start_idx:end_idx])
        baseline_values.append(average_intensity)

    coefficients = np.polyfit(fixed_points, baseline_values, poly_order)
    polynomial = np.poly1d(coefficients)
    baseline = polynomial(waves)
    intensities_corrected = intensities - baseline

    return intensities_corrected

def preprocess_raman_file(file_path, pca_comp):
    """
    Processes the Raman spectroscopy data file.
    """
    # Read the file into a dataframe
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)
    df.columns = ['X', 'Y', 'Wave', 'Intensity']

    # Adjust X and Y values so that the smallest value is 0
    min_x = df['X'].min()
    min_y = df['Y'].min()
    df['X'] = df['X'] - min_x
    df['Y'] = df['Y'] - min_y

    # Pivot the dataframe to get the intensity matrix
    pivot_table = df.pivot_table(index=['X', 'Y'], columns='Wave', values='Intensity', fill_value=0)
    waves = pivot_table.columns.values
    intensity_matrix = pivot_table.values

    # Perform PCA to reduce noise
    pca = PCA(n_components=pca_comp)
    intensity_matrix_pca = pca.fit_transform(intensity_matrix)
    intensity_matrix_reconstructed = pca.inverse_transform(intensity_matrix_pca)

    # Convert the reconstructed intensity matrix back to a dataframe
    df_reconstructed = pd.DataFrame(intensity_matrix_reconstructed, index=pivot_table.index, columns=pivot_table.columns)

    # Apply baseline correction to each spectrum
    df_reconstructed_corrected = pd.DataFrame()
    for (x, y), spectrum in df_reconstructed.iterrows():
        corrected_spectrum = spectrum.values
        corrected_spectrum = baseline_correction(waves, corrected_spectrum, fixed_points=[500, 600, 700, 900, 1850, 1900])
        corrected_spectrum = savgol_filter(corrected_spectrum, window_length=33, polyorder=1)
        corrected_spectrum_df = pd.DataFrame(corrected_spectrum, index=waves, columns=['Intensity'])
        corrected_spectrum_df['X'] = x
        corrected_spectrum_df['Y'] = y
        df_reconstructed_corrected = pd.concat([df_reconstructed_corrected, corrected_spectrum_df.reset_index()])

    # Rename columns to match the original structure
    df_reconstructed_corrected.columns = ['Wave', 'Intensity', 'X', 'Y']

    # Create the new file path
    base, ext = os.path.splitext(file_path)
    output_dir = f"Fittings_{os.path.basename(base)}"
    os.makedirs(output_dir, exist_ok=True)
    new_file_path = os.path.join(output_dir, f"{os.path.basename(base)}_fixed{ext}")

    # Save the processed dataframe to a new file
    df_reconstructed_corrected.to_csv(new_file_path, sep='\t', index=False)

    return df_reconstructed_corrected, new_file_path

