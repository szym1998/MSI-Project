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

#clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

def plot_time_domain(wav_obj, channel=0, dpi=100, line_width=0.5):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj
    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    print("Creating time axis...")
    # Create a time axis for the data
    start_time = time.time()
    time_axis = np.arange(len(data_channel)) / float(sample_rate)
    print(f"Time axis creation took {time.time() - start_time:.2f} seconds")

    print("Creating plot...")
    # Create a figure and plot the data
    start_time = time.time()
    fig, ax = plt.subplots()
    ax.plot(time_axis, data_channel, color='black', linewidth=line_width)
    print(f"Plot creation took {time.time() - start_time:.2f} seconds")

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    # Set the line width and DPI of the plot
    for l in ax.lines:
        l.set_linewidth(line_width)
    fig.set_dpi(dpi)

    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf



def plot_frequency_domain(wav_obj, channel=0, dpi=100, n_fft=2048, hop_length=512):
    # Get the sample rate and data from the WAV object
    sample_rate, data = wav_obj
    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    print("Computing STFT...")
    # Compute the STFT of the data
    start_time = time.time()
    f, t, Zxx = signal.stft(data_channel, sample_rate, nperseg=n_fft, noverlap=hop_length)
    print(f"STFT computation took {time.time() - start_time:.2f} seconds")

    print("Creating plot...")
    # Create a figure and plot the STFT
    start_time = time.time()
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(t, f, np.abs(Zxx), cmap='viridis', shading='gouraud')
    print(f"Plot creation took {time.time() - start_time:.2f} seconds")

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Set the colorbar
    fig.colorbar(pcm, ax=ax)

    # Set the DPI of the plot
    fig.set_dpi(dpi)

    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf


def plot_cwt(wav_obj, channel=0, dpi=100, wavelet='morl', scales=np.arange(1, 400)):
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
    fig.set_dpi(dpi)

    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf


def plot_fcwt(wav_obj, channel=0, dpi=100, f0=0.1, f1=5, fn=100):
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
    fig.set_dpi(dpi)
    
    print("Converting plot to BytesIO object...")
    # Convert the plot to a BytesIO object
    start_time = time.time()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)

    plt.close(fig)
    print(f"Plot conversion took {time.time() - start_time:.2f} seconds")

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object
    return buf



import os
from pathlib import Path

# Define the input file path and output folder
input_file = Path('/home/szymon/coding/MSI_Project/output_wav_files/sine3.wav')
output_folder = 'output_plots'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the WAV file
sample_rate, data = wavfile.read(input_file)



# Plot the time domain waveform and save the plot to a file
time_plot_file = os.path.join(output_folder, 'time_domain.png')
time_plot = plot_time_domain((sample_rate, data))
with open(time_plot_file, 'wb') as f:
    f.write(time_plot.getbuffer())

# Plot the frequency domain spectrogram and save the plot to a file
freq_plot_file = os.path.join(output_folder, 'frequency_domain.png')
freq_plot = plot_frequency_domain((sample_rate, data))
with open(freq_plot_file, 'wb') as f:
    f.write(freq_plot.getbuffer())

# Plot the CWT and save the plot to a file
cwt_plot_file = os.path.join(output_folder, 'cwt.png')
cwt_plot = plot_cwt((sample_rate, data))
with open(cwt_plot_file, 'wb') as f:
    f.write(cwt_plot.getbuffer())


# Plot the FCWT and save the plot to a file
fcwt_plot_file = os.path.join(output_folder, 'fcwt.png')
fcwt_plot = plot_fcwt((sample_rate, data))
with open(fcwt_plot_file, 'wb') as f:
    f.write(fcwt_plot.getbuffer())