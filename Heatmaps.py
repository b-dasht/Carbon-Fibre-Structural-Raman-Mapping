import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    # Access the colour bar and configure it
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
    cbar_width = 0.8  # Width of the colour bar
    cbar_padding = 0.2  # Padding between the heatmap and colour bar
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