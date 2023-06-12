import io
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.io import wavfile
import os
from pathlib import Path
from scipy import signal
import time 
import fcwt
from convert_to_wav import x2w
#clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
import matplotlib.pyplot as plt
import io
import time

def plot_time_domain(wav_obj, channel=0, line_width=0.5, xlim=None, ylim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj

    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    # Create a time axis for the data
    time_axis = np.arange(len(data_channel)) / float(sample_rate)

    # Create a figure and plot the data
    fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the size as needed
    ax.plot(time_axis, data_channel, color='black', linewidth=line_width)

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    # Set default xlim and ylim if not provided
    if not xlim:
        xlim = [0, len(data_channel) / float(sample_rate)]
    if not ylim:
        min_data = np.min(data_channel)
        max_data = np.max(data_channel)
        ylim = [min_data*1.1, max_data*1.1]

    #limit xlim and ylim to 2 decimal places
    xlim = [round(x, 2) for x in xlim]
    ylim = [round(y, 2) for y in ylim]

    # Set xlim and ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the line width
    for l in ax.lines:
        l.set_linewidth(line_width)

    # Tight layout
    plt.tight_layout()

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the default xlim and ylim values
    return buf, xlim, ylim





def plot_frequency_domain(wav_obj, channel=0, dpi=100, n_fft=2048, hop_length=512, xlim=None, ylim=None, cbar_lim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj

    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    # Compute the STFT of the data
    f, t, Zxx = signal.stft(data_channel, sample_rate, nperseg=n_fft, noverlap=hop_length)

    # Calculate the original xlim, ylim, and cbar_lim
    original_xlim = [0, t[-1]]
    original_ylim = [0, f[-1]]
    original_cbar_lim = [np.abs(Zxx).min(), np.abs(Zxx).max()]

    # Create a figure and plot the STFT
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    fig.set_dpi(dpi)

    # Use the original or provided cbar_lim for the color map
    cbar_lim = cbar_lim or original_cbar_lim
    pcm = ax.pcolormesh(t, f, np.abs(Zxx), cmap='viridis', shading='gouraud', rasterized=True, vmin=cbar_lim[0], vmax=cbar_lim[1])

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Use the original or provided xlim and ylim
    xlim = xlim or original_xlim
    ylim = ylim or original_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the colorbar
    fig.colorbar(pcm, ax=ax)

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the original xlim, ylim, and cbar_lim values
    return buf, original_xlim, original_ylim, original_cbar_lim





def plot_cwt(wav_obj, channel=0, wavelet='morl', scales=np.arange(1, 400)):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj
    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    print("Computing CWT...")
    # Compute the CWT of the data
    start_time = time.time()
    coefs, freqs = pywt.cwt(data_channel, scales, wavelet)
    print(f"CWT computation took {time.time() - start_time:.2f} seconds")

    print("Creating plot...")
    # Create a figure and plot the CWT
    start_time = time.time()
    fig, ax = plt.subplots()
    cwt_img = ax.imshow(np.abs(coefs), extent=[0, len(data_channel) / float(sample_rate), freqs[-1], freqs[0]], cmap='viridis', aspect='auto')
    print(f"Plot creation took {time.time() - start_time:.2f} seconds")

    # Set the x and y labels
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')

    # Set the colorbar
    fig.colorbar(cwt_img, ax=ax)

    # Set the DPI of the plot
    

    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf


def plot_fcwt(wav_obj, channel=0, f0=0.1, f1=5, fn=100):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj
    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    print("Computing FCWT...")
    # Compute the FCWT of the data
    start_time = time.time()
    morl = fcwt.Morlet(2.0)
    scales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, sample_rate, f0, f1, fn)
    nthreads = 8
    use_optimization_plan = False
    use_normalization = True
    fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)
    output = np.zeros((fn, data_channel.size), dtype=np.complex64)
    fcwt_obj.cwt(data_channel, scales, output)
    print(f"FCWT computation took {time.time() - start_time:.2f} seconds")

    print("Creating plot...")
    # Create a figure and plot the FCWT
    start_time = time.time()
    fig, ax = plt.subplots()

    cwt_img = ax.imshow(np.abs(output), aspect='auto')
    
    print(f"Plot creation took {time.time() - start_time:.2f} seconds")

    # Set the x and y labels
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')

    # Set the colorbar
    fig.colorbar(cwt_img, ax=ax)

    # Set the DPI of the plot
    
    
    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')

    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf



import os
from pathlib import Path

# Define the input file path and output folder
input_file = Path('/home/szymon/coding/MSI_Project/datastore/borg.wav') 

conv_file = Path('/home/szymon/coding/MSI_Project/datastore/borg2.wav')

x2w(input_file, conv_file)

output_folder = 'output_plots'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the WAV file

sample_rate, data = wavfile.read(conv_file)



# # Plot the time domain waveform and save the plot to a file
# time_plot_file = os.path.join(output_folder, 'time_domain.svg')
# time_plot, xlim, ylim = plot_time_domain((sample_rate, data),xlim=(0, 0.02), channel=1)
# print(xlim)
# print(ylim)
# with open(time_plot_file, 'wb') as f:
#     f.write(time_plot.getbuffer())

# Plot the frequency domain spectrogram and save the plot to a file
freq_plot_file = os.path.join(output_folder, 'frequency_domain.svg')
freq_plot, xlim, ylim, clim = plot_frequency_domain((sample_rate, data), xlim=(4, 8), ylim=(0, 4000), cbar_lim=(0.1, 0.2))
print(xlim)
print(ylim)
print(clim)
with open(freq_plot_file, 'wb') as f:
    f.write(freq_plot.getbuffer())

# # Plot the CWT and save the plot to a file
# cwt_plot_file = os.path.join(output_folder, 'cwt.svg')
# cwt_plot = plot_cwt((sample_rate, data))
# with open(cwt_plot_file, 'wb') as f:
#     f.write(cwt_plot.getbuffer())


# # Plot the FCWT and save the plot to a file
# fcwt_plot_file = os.path.join(output_folder, 'fcwt.svg')
# fcwt_plot = plot_fcwt((sample_rate, data))
# with open(fcwt_plot_file, 'wb') as f:
#     f.write(fcwt_plot.getbuffer())