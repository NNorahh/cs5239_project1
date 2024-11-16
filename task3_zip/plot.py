#!/usr/bin/python3

import matplotlib.pyplot as plt


def main():
    # Compression levels
    level = list(range(10))

    # Fill in real_time (Time Taken in ms)
    real_time = [254, 410, 445, 520, 542, 749, 952, 1111, 1650, 1480]  # Time in ms (converted from TimeTaken)

    # Fill in archive_size (CompressedSize in bytes)
    archive_size = [11908, 5628, 5440, 5275, 5184, 5065, 5010, 4999, 4992, 4992]  # Size in bytes

    # Convert archive_size to kilobytes for better readability
    archive_size = [it / 1024 for it in archive_size]

    # Create subplots
    fig, axs = plt.subplots(2)

    # Plot time taken (in ms)
    axs[0].plot(level, real_time, 'bo-')
    axs[0].set_title('Compression Time vs Compression Level')

    # Plot archive size (in KB)
    axs[1].plot(level, archive_size, 'ro-')
    axs[1].set_title('Output Archive Size vs Compression Level')

    # Set x-axis ticks to match compression level (0-9)
    axs[0].set_xticks(level)
    axs[1].set_xticks(level)

    # Enable grid for both subplots
    axs[0].grid(True, which='both')
    axs[1].grid(True, which='both')

    # Set labels
    axs[0].set_xlabel('Compression level')
    axs[1].set_xlabel('Compression level')

    axs[0].set_ylabel('Time [ms]')
    axs[1].set_ylabel('Archive size [KB]')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
