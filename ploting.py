import io
import base64
import numpy as np
import pywt
import os
from pathlib import Path
from scipy import signal
import fcwt
from convert_to_wav import x2w

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_td(path_to_file, xlim=None, ylim=None):
    channel=0
    line_width=0.5
    dpi = 100
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)
    og_xlim = (0, len(data) / sample_rate)
    og_ylim = [-1, 1]

    

    if xlim == [None] or None:        
        xlim = og_xlim
    if ylim == [None] or None:
        ylim = og_ylim

    
    data_channel = data
    
    
    #normalize data
    data_channel = data_channel / np.max(np.abs(data_channel))

    # Create a time axis for the data
    time_axis = np.arange(len(data_channel)) / float(sample_rate)

    # Create a figure and plot the data
    fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the size as needed
    ax.plot(time_axis, data_channel, color='black', linewidth=line_width)

    # Set the x and y labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    # Set the title
    ax.set_title('Time Domain')

    # Set default xlim and ylim if not provided
    if not xlim:
        xlim = [0, len(data_channel) / float(sample_rate)]
    if not ylim:
        min_data = np.min(data_channel)
        max_data = np.max(data_channel)
        ylim = [min_data*1.1, max_data*1.1]
    
    # Store original xlim and ylim
    
    

    

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

    image_string = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Return the base64 string along with the original xlim, ylim, and clim values
    result = {
        "image_string_td": image_string,
        "xlim_td": og_xlim,
        "ylim_td": og_ylim,
        "xlim_td_current": xlim,
        "ylim_td_current": ylim,
    }

    return result


def plot_stft(path_to_file, n_fft=None, hop_length=None, xlim=None, ylim=None, cbar_lim=None):
    dpi=100

    
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    if len(data) < 5000:
        og_n_fft = sample_rate // 20
    else:
        og_n_fft = sample_rate // 200

    if len(data) < 5000:
        og_hop_length = og_n_fft // 1.1
    else:
        og_hop_length = og_n_fft // 2

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

    if hop_length > n_fft:
        hop_length = n_fft // 1.1

    
    data_channel = data
    

    # Compute the STFT of the data
    f, t, Zxx = signal.stft(
        data_channel, sample_rate, nperseg=n_fft, noverlap=hop_length
    )

    # Calculate the original xlim, ylim, and cbar_lim
    og_xlim = [0, t[-1]]
    og_ylim = [0, f[-1]]
    og_clim = [np.abs(Zxx).min(), np.abs(Zxx).max()]

    # Create a figure and plot the STFT
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    fig.set_dpi(dpi)

    # Use the original or provided cbar_lim for the color map
    cbar_lim = cbar_lim or og_clim
    pcm = ax.pcolormesh(
        t,
        f,
        np.abs(Zxx),
        cmap="viridis",
        shading="gouraud",
        rasterized=True,
        vmin=cbar_lim[0],
        vmax=cbar_lim[1],
    )

    # Set the x and y labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    # Set the title
    ax.set_title("STFT")

    # Use the original or provided xlim and ylim
    xlim = xlim or og_xlim
    ylim = ylim or og_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the colorbar
    fig.colorbar(pcm, ax=ax)
    plt.tight_layout()

    # Convert the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="svg", dpi=dpi)
    plt.close(fig)

    # Reset the buffer's position to the start
    buf.seek(0)

    image_string = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Return the base64 string along with the original xlim, ylim, and clim values
    result = {
        "image_string_stft": image_string,
        "xlim_stft": og_xlim,
        "ylim_stft": og_ylim,
        "clim_stft": og_clim,
        "nfft_stft": og_n_fft,
        "hop_length_stft": og_hop_length,
    }

    return result


def plot_fcwt(path_to_file, f0=None, f1=None, fn=None, mor_size=None, xlim=None, ylim=None, clim=None):
    dpi=100
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    
    og_f1 = sample_rate // 2

    if f1 is None:                                            
        f1 = sample_rate // 2
        
    if fn is None:
        #if len data shorter than 5000 samples, use 1/10 of the data length
        if len(data) < 5000:
            fn = len(data)//10
            original_fn = len(data)//10
        else:
            fn = len(data)//500
            original_fn = len(data)//500
            

    if mor_size is None:
        mor_size = sample_rate//1000
    if f0 is None:
        f0 = 1

    
    data_channel = data
    

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
    # Set the title
    ax.set_title('CWT')

    # Get the original xlim and ylim
    og_xlim = ax.get_xlim()
    og_ylim = ax.get_ylim()

    # Set the xlim and ylim if they are provided
    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # Set the colorbar
    cbar = fig.colorbar(cwt_img, ax=ax)

    # Get the original clim
    og_clim = cwt_img.get_clim()
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

    image_string = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Return the base64 string along with the original xlim, ylim, and clim values
    result = {
        "image_string_fcwt": image_string,
        "xlim_fcwt": og_xlim,
        "ylim_fcwt": og_ylim,
        "clim_fcwt": og_clim,
        "f1_fcwt": og_f1,
    }

    return result


def plot_fft(path_to_file, xlim=None, ylim=None):
    dpi=100
    # Get the sample rate and data from the WAV object
    sample_rate, data = x2w(path_to_file)

    # If the data array is one-dimensional, set data_channel to data
    
    data_channel = data
    

    # Compute the FFT of the data
    freqs = np.fft.rfftfreq(len(data_channel), 1/sample_rate)
    fft = np.fft.rfft(data_channel)

    # Create a figure and plot the FFT
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    ax.plot(freqs, np.abs(fft), color='black')

    # Set the x and y labels
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    # Set the title
    ax.set_title('FFT')

    # Get the original xlim and ylim
    og_xlim = ax.get_xlim()
    og_ylim = ax.get_ylim()

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

    # Convert the BytesIO object to a base64 string
    image_string = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Return the base64 string along with the original xlim and ylim values
    result = {
        "image_string_fft": image_string,
        "xlim_fft": og_xlim,
        "ylim_fft": og_ylim
    }

    return result


