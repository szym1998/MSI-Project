# test_conversion.py

import os
from convert_to_wav import convert_to_wav
import numpy as np

input_files = ['file_example_MP3_700KB1.csv', 'file_example_MP3_700KB2.mp3', 'file_example_MP3_700KB3.wav']

output_folder = 'output_wav_files'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sampling_rate = 32000  # You can set the desired sampling rate here

for input_file in input_files:
    output_file = os.path.join(output_folder, os.path.splitext(input_file)[0] + '.wav')
    try:
        print(f"Converting {input_file} to {output_file}")
        convert_to_wav(('files/'+input_file), output_file, sampling_rate)
        print(f"Conversion successful")
    except Exception as e:
        print(f"Conversion failed: {e}")




import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_wav_files(output_files, plot_file, dpi=300, line_width=0.25):
    # create a figure to plot the WAV files
    fig, ax = plt.subplots()

    for i, file in enumerate(output_files):
        # Open the output file using scipy.io.wavfile.read
        # This returns the sample rate and the data array
        sample_rate, data = wavfile.read(file)

        # Get the duration of the file by dividing the length of the data by the sample rate
        duration = len(data) / float(sample_rate)

        # Get the number of channels in the data array
        num_channels = data.shape[1] if len(data.shape) > 1 else 1

        # Create a time axis for the data
        time = 1/float(sample_rate) * np.arange(len(data))

        # Plot the data on the same figure with different colors
        ax.plot(time, data[:, 0], color=f'C{i}', label=file, linewidth=line_width)

        # Print the sample rate, duration, and number of channels
        print(f"File: {file}")
        print(f"Sampling rate: {sample_rate}")
        print(f"Duration: {duration} seconds")
        print(f"Number of channels: {num_channels}")

    # Set the legend
    ax.legend()

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    # Set the line width and DPI of the plot
    for l in ax.lines:
        l.set_linewidth(line_width)
    fig.set_dpi(dpi)

    # Save the plot to a PNG file
    plt.savefig(plot_file, dpi=dpi)

    # Close the figure
    plt.close(fig)


#plot now
output_files = ['output_wav_files/file_example_MP3_700KB1.wav', 'output_wav_files/file_example_MP3_700KB2.wav', 'output_wav_files/file_example_MP3_700KB3.wav']
plot_file = 'wav_files.png'

plot_wav_files(output_files, plot_file)