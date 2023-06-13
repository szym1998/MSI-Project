import io
import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
from pathlib import Path
from scipy import signal
import fcwt
from convert_to_wav import x2w


def plot_time_domain(path_to_file, channel=0, line_width=0.5, xlim=None, ylim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

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


def plot_frequency_domain(path_to_file, channel=0, dpi=100, n_fft=None, hop_length=None, xlim=None, ylim=None, cbar_lim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    if not n_fft:
        if len(data) < 5000:
            n_fft = sample_rate // 20
        else:
            n_fft = sample_rate // 200
    if not hop_length:
        if len(data) < 5000:
            hop_length = n_fft // 1.1
        else:
            hop_length = n_fft // 2

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
    plt.tight_layout()

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the original xlim, ylim, and cbar_lim values
    return buf, original_xlim, original_ylim, original_cbar_lim


def plot_cwt(path_to_file, channel=0, wavelet='morl', scales=np.arange(15000, 20000), dpi=100, xlim=None, ylim=None, cbar_lim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    # Compute the CWT of the data
    coefs, freqs = pywt.cwt(data_channel, scales, wavelet)

    # Calculate the original xlim, ylim, and cbar_lim
    original_xlim = [0, len(data_channel) / float(sample_rate)]
    original_ylim = [freqs[0], freqs[-1]]
    original_cbar_lim = [np.abs(coefs).min(), np.abs(coefs).max()]

    # Create a figure and plot the CWT
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_dpi(dpi)

    # Use the original or provided cbar_lim for the color map
    cbar_lim = cbar_lim or original_cbar_lim
    cwt_img = ax.imshow(np.abs(coefs), extent=[0, len(data_channel) / float(sample_rate), freqs[-1], freqs[0]], cmap='viridis', aspect='auto', vmin=cbar_lim[0], vmax=cbar_lim[1])

    # Set the x and y labels
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')

    # Use the original or provided xlim and ylim
    xlim = xlim or original_xlim
    ylim = ylim or original_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the colorbar
    fig.colorbar(cwt_img, ax=ax)

    plt.tight_layout()

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the original xlim, ylim, and cbar_lim values
    return buf, original_xlim, original_ylim, original_cbar_lim


def plot_fcwt(path_to_file, channel=0, f0=None, f1=None, fn=None, dpi=100, mor_size=None, xlim=None, ylim=None, clim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    if f1 is None:                                            
        f1 = sample_rate // 2
    if fn is None:
        #if len data shorter than 5000 samples, use 1/10 of the data length
        if len(data) < 5000:
            fn = len(data)//10
        else:
            fn = len(data)//500
            

    if mor_size is None:
        mor_size = sample_rate//1000
    if f0 is None:
        f0 = 1

    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    # Compute the FCWT of the data
    morl = fcwt.Morlet(mor_size)
    scales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, sample_rate, f0, f1, fn)
    nthreads = 8
    use_optimization_plan = False
    use_normalization = True
    fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)
    output = np.zeros((fn, data_channel.size), dtype=np.complex64)
    fcwt_obj.cwt(data_channel, scales, output)

    # Create a figure and plot the FCWT
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    fig.set_dpi(dpi)
    time_extent = [0, len(data_channel) / float(sample_rate)]
    cwt_img = ax.imshow(np.abs(output), extent=[*time_extent, f0, f1], aspect='auto', cmap='viridis')

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Get the original xlim and ylim
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    # Set the xlim and ylim if they are provided
    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # Set the colorbar
    cbar = fig.colorbar(cwt_img, ax=ax)

    # Get the original clim
    original_clim = cwt_img.get_clim()
    plt.tight_layout()

    # Set the color limits if they are provided
    if clim:
        cwt_img.set_clim(clim)

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the original xlim, ylim and clim values
    return buf, original_xlim, original_ylim, original_clim


def plot_fft(path_to_file, channel=0, dpi=100, xlim=None, ylim=None):
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    # If the data array is one-dimensional, set data_channel to data
    if data.ndim == 1:
        data_channel = data
    # Otherwise, extract the specified channel from the data array
    else:
        data_channel = data[:, channel]

    # Compute the FFT of the data
    freqs = np.fft.rfftfreq(len(data_channel), 1/sample_rate)
    fft = np.fft.rfft(data_channel)

    # Create a figure and plot the FFT
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    ax.plot(freqs, np.abs(fft), color='black')

    # Set the x and y labels
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')

    # Get the original xlim and ylim
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    # Set the xlim and ylim if they are provided
    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    # Return the BytesIO object along with the original xlim and ylim values
    return buf, original_xlim, original_ylim



# import os
# from pathlib import Path

# # Define the input file path and output folder
# input_file = Path('/home/szymon/coding/MSI_Project/datastore/borg.wav') 

# output_folder = 'output_plots'

# # Load the WAV file
# sample_rate, data = x2w(input_file)



# # Plot the time domain waveform and save the plot to a file
# time_plot_file = os.path.join(output_folder, 'time_domain.svg')
# time_plot, xlim, ylim = plot_time_domain((sample_rate, data))
# print(xlim)
# print(ylim)
# with open(time_plot_file, 'wb') as f:
#     f.write(time_plot.getbuffer())

# # Plot the frequency domain spectrogram and save the plot to a file
# freq_plot_file = os.path.join(output_folder, 'frequency_domain.svg')
# freq_plot, xlim, ylim, clim = plot_frequency_domain((sample_rate, data))
# print(xlim)
# print(ylim)
# print(clim)
# with open(freq_plot_file, 'wb') as f:
#     f.write(freq_plot.getbuffer())

# # # Plot the CWT and save the plot to a file
# # cwt_plot_file = os.path.join(output_folder, 'cwt.svg')
# # cwt_plot, xlim, ylim, clim  = plot_cwt((sample_rate, data))
# # print(xlim)
# # print(ylim)
# # print(clim)
# # with open(cwt_plot_file, 'wb') as f:
# #     f.write(cwt_plot.getbuffer())


# # Plot the FCWT and save the plot to a file
# fcwt_plot_file = os.path.join(output_folder, 'fcwt.svg')
# fcwt_plot, xlim, ylim, clim  = plot_fcwt((sample_rate, data), dpi=100)
# print(xlim)
# print(ylim)
# print(clim)
# with open(fcwt_plot_file, 'wb') as f:
#     f.write(fcwt_plot.getbuffer())


# # Plot the FFT and save the plot to a file
# fft_plot_file = os.path.join(output_folder, 'fft.svg')
# fft_plot, xlim, ylim = plot_fft((sample_rate, data), dpi=100)
# print(xlim)
# print(ylim)
# with open(fft_plot_file, 'wb') as f:
#     f.write(fft_plot.getbuffer())
