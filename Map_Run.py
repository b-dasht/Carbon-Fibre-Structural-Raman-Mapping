import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from Preprocess_RamanMap import process_raman_file
from Extract_Spectra import extract_spectra
from Single_Scan_Model import fit_and_save_spectra
from BaselineCorrection import baseline_correction
from Spectra_Plots import plot_spectra


# Define the input file
input_files = ['Pristine.txt', 'Desized.txt','ASPN.txt']

for file in input_files:

    input_file = file

    # Define the output directory
    output_dir = f"Fittings_{input_file[:-4]}"
    os.makedirs(output_dir, exist_ok=True)


    # Process the file and perform PCA to denoise the data
    overall_data = process_raman_file(input_file, 5)

    # Initialise new columns in overall_data
    new_columns = ['Smoothed Intensity', 'Baseline', 'Baseline Corrected', 'D Band Fit', 'G Band Fit', 'D2 Band Fit', 'G2 Band Fit', 'Combined Fit']
    for col in new_columns:
        overall_data[col] = np.nan

    # Determine the maximum X and Y values
    max_x = int(overall_data['X'].max())
    max_y = int(overall_data['Y'].max())

    # Initialise results list
    results = []

    # Loop through all X and Y combinations
    for x_val in range(max_x + 1):
        for y_val in range(max_y + 1):
            # Extract spectra for the current X and Y combination
            waves, raw_intensities, denoised_intensities = extract_spectra(overall_data, x_val, y_val, 'Raw Intensity', 'Denoised Intensity')

            # Smooth data
            smoothed_intensities = savgol_filter(denoised_intensities, window_length=27, polyorder=1)

            # Baseline correction
            corrected_intensities, baseline = baseline_correction(waves, smoothed_intensities, polyorder=2, numstd=0.4)

            # Perform the fitting
            id_fit, ig_fit, id2_fit, ig2_fit, combined_fit, r_squared, ratio_id_ig_intensity, ratio_id_ig_area = fit_and_save_spectra(waves, corrected_intensities, x_val, y_val)

            # Define a dictionary mapping column names to their corresponding data arrays
            fit_results = {
                'Smoothed Intensity': smoothed_intensities,
                'Baseline': baseline,
                'Baseline Corrected': corrected_intensities,
                'D Band Fit': id_fit,
                'G Band Fit': ig_fit,
                'D2 Band Fit': id2_fit,
                'G2 Band Fit': ig2_fit,
                'Combined Fit': combined_fit
            }

            # Extract the wavenumber values for interpolation
            wave_values = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Wave']

            # Update the dataframe with results
            for col, result_array in fit_results.items():
                overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), col] = np.interp(wave_values, waves, result_array)

            # Store results
            results.append([x_val, y_val, r_squared, ratio_id_ig_intensity, ratio_id_ig_area])

            # Save data and fitting results to text files
            output_file = os.path.join(output_dir, f"FittingResults-{x_val},{y_val}.txt")
            with open(output_file, 'w') as f:
                f.write("Wavenumber\tRaw Intensity\tDenoised Intensity\tSmoothed Intensity\tBaseline\tBaseline Corrected\tD Band Fit\tG Band Fit\tD2 Band Fit\tG2 Band Fit\tCombined Fit\n")
                for wave, raw_intensity, denoised_intensity, smoothed_intensity, baseline_value, corrected_intensity, id_fit_value, ig_fit_value, id2_fit_value, ig2_fit_value, combined_fit_value in zip(
                        waves, raw_intensities, denoised_intensities, smoothed_intensities,
                        baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit):
                    f.write(f"{wave:.2f}\t{raw_intensity:.2f}\t{denoised_intensity:.2f}\t{smoothed_intensity:.2f}\t{baseline_value:.2f}\t{corrected_intensity:.2f}\t{id_fit_value:.2f}\t{ig_fit_value:.2f}\t{id2_fit_value:.2f}\t{ig2_fit_value:.2f}\t{combined_fit_value:.2f}\n")

            # Print progress
            print(f"Completed fitting and saving for X={x_val}, Y={y_val}")


    # Create dataframe from results
    results_data = pd.DataFrame(results, columns=['X', 'Y', 'R-Squared', 'ID/IG Intensity', 'ID/IG Area'])

    # Define the output file name for results
    output_file = os.path.join(output_dir, f"{input_file[:-4]}_ratios-and-fit-quality.txt")

    # Save the Results table
    results_data.to_csv(output_file, index=False, sep='\t')

    # Save overall data
    overall_data_output_file = os.path.join(output_dir, f"{input_file[:-4]}overall-processed-data.txt")
    overall_data.to_csv(overall_data_output_file, index=False, sep='\t')

    # Loop through all X and Y combinations to create plots
    for x_val in range(max_x + 1):
        for y_val in range(max_y + 1):
            # Extract spectra for the current X and Y combination
            waves = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Wave']
            raw = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Raw Intensity']
            denoised = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Denoised Intensity']
            smoothed_intensities = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Smoothed Intensity']
            baseline = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Baseline']
            corrected_intensities = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Baseline Corrected']
            id_fit = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'D Band Fit']
            ig_fit = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'G Band Fit']
            id2_fit = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'D2 Band Fit']
            ig2_fit = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'G2 Band Fit']
            combined_fit = overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Combined Fit']

            # Plot spectra
            plot_spectra(output_dir, x_val, y_val, waves, raw, denoised, smoothed_intensities, baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit)

