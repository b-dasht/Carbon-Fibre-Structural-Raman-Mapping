import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_two_spectra(name_1, x1_data, name_2, y1_data, x2_data, y2_data):
    """
    Plots the original and reconstructed spectra for a single spectrum.

    Parameters:
    - x1_data: Array or list of spectra 1 x-axis data points.
    - y1_data: Array or list of spectra 1 y-axis data points.
    - x2_data: Array or list of spectra 2 x-axis data points.
    - y2_data: Array or list of spectra 2 y-axis data points.
    """

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot spectra 1
    plt.plot(x1_data, y1_data, label=str(name_1), color='blue', alpha=0.7)

    # Plot spectra 2
    plt.plot(x2_data, y2_data, label=str(name_2), color='red', alpha=0.7)

    # Add titles and labels
    plt.xlabel('Wave')
    plt.ylabel('Intensity')

    # Display the legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_three_spectra(name_1, x1_data, y1_data, name_2, x2_data, y2_data, name_3, x3_data, y3_data):
    """
    Plots the original and reconstructed spectra for a single spectrum.

    Parameters:
    - x1_data: Array or list of spectra 1 x-axis data points.
    - y1_data: Array or list of spectra 1 y-axis data points.
    - x2_data: Array or list of spectra 2 x-axis data points.
    - y2_data: Array or list of spectra 2 y-axis data points.
    - x3_data: Array or list of spectra 3 x-axis data points.
    - y3_data: Array or list of spectra 3 y-axis data points.
    """

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot spectra 1
    plt.plot(x1_data, y1_data, label=str(name_1), color='blue', alpha=0.7)

    # Plot spectra 2
    plt.plot(x2_data, y2_data, label=str(name_2), color='red', alpha=0.7)

    # Plot spectra 3
    plt.plot(x3_data, y3_data, label=str(name_3), color='green', alpha=0.7)

    # Add titles and labels
    plt.xlabel('Wave')
    plt.ylabel('Intensity')

    # Display the legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_heatmap(data, title, file_name, max_x, max_y, v_min, v_max):
    plt.figure(figsize=(2, 20))
    ax = sns.heatmap(
        data,
        cmap='viridis',
        annot=False,
        vmin=v_min,  # Minimum value of the colour bar
        vmax=v_max,  # Maximum value of the colour bar
        cbar_kws={'label': 'ID/IG Ratio Colour Scale', 'orientation': 'vertical'}
    )

    # Set up the colour bar
    cbar = ax.collections[0].colorbar
    cbar.set_label('ID/IG Ratio Colour Scale', fontsize=25, fontweight='bold')

    # Set colour bar ticks and labels
    num_ticks = max(10, int((v_max - v_min) / 0.1))  # Ensure at least 2 ticks
    tick_values = np.linspace(v_min, v_max, num=num_ticks)
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels([f'{label:.2f}' for label in tick_values])
    cbar.ax.tick_params(labelsize=25)

    # Set x and y limits to match the size of the matrix
    ax.set_xlim(0, max_y)
    ax.set_ylim(max_x, 0)

    # Adjust the colour bar size and position to stretch the entire length of the heatmap
    pos = ax.get_position()
    cbar_width = 0.8
    cbar_padding = 0.2
    cbar_position = [
        pos.x1 + cbar_padding,  # Left position
        pos.y0,  # Bottom position
        cbar_width,  # Width
        pos.height  # Height (stretch to match heatmap height)
    ]
    cbar.ax.set_position(cbar_position)

    # Centre title
    plt.title(title, fontsize=25, fontweight='bold', loc='center', y=1.02)

    # Increase the font size of the labels and markers
    plt.xlabel('X', fontsize=25, fontweight='bold')
    plt.ylabel('Y', fontsize=25, fontweight='bold')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))  # X-axis tick intervals of 1
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))  # Y-axis tick intervals of 5

    # Set the tick positions and labels
    xticks = np.arange(0, max_y + 1, 1)
    yticks = np.arange(0, max_x + 1, 5)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{i:.0f}' for i in xticks])
    ax.set_yticklabels([f'{i:.1f}' for i in yticks])
    ax.tick_params(axis='both', which='major', labelsize=25)

    # Save the figure with the colour bar aligned to the heatmap
    plt.savefig(file_name, dpi=300)







# Define a function to plot spectra
def plot_spectra(output_dir, x_val, y_val, waves, raw, denoised, smoothed_intensities, baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit):
    plt.figure(figsize=(12, 8))

    # Plot raw intensity
    plt.plot(waves, raw, label='Raw Data', color='black')

    # Plot denoised intensity
    plt.plot(waves, denoised, label='Denoised', color='red')

    # Plot smoothed intensity
    plt.plot(waves, smoothed_intensities, label='Smoothed', color='blue')

    # Plot baseline
    plt.plot(waves, baseline, label='Baseline', color='green')

    # Plot baseline-corrected intensity
    plt.plot(waves, corrected_intensities, label='Baseline Corrected', color='black')

    # Plot fitting results
    plt.plot(waves, id_fit, label='D Band Fit', color='purple')
    plt.plot(waves, ig_fit, label='G Band Fit', color='orange')
    plt.plot(waves, id2_fit, label='D2 Band Fit', color='pink')
    plt.plot(waves, ig2_fit, label='G2 Band Fit', color='brown')
    plt.plot(waves, combined_fit, label='Combined Fit', color='red')

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Spectra for X={x_val}, Y={y_val}')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_file = os.path.join(output_dir, f'Fitted-Spectra_{x_val},{y_val}.png')
    plt.savefig(plot_file)
    plt.close()
