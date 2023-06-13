import os
import numpy as np
import pandas as pd
from scipy.io import wavfile, loadmat
from pydub import AudioSegment



def x2w(input_file, sampling_rate=None):
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext in {'.csv', '.txt'}:
        data = pd.read_csv(input_file, header=None).values
        
    elif file_ext == '.xlsx':
        data = pd.read_excel(input_file, header=None).values
    elif file_ext == '.mat':
        mat = loadmat(input_file)
        data = mat[list(mat.keys())[-1]]
    elif file_ext == '.wav':
        try:
            _, data = wavfile.read(input_file)
            if sampling_rate is None:
                sampling_rate = wavfile.read(input_file)[0]
        except ValueError:
            audio = AudioSegment.from_file(input_file)
            if sampling_rate is None:
                sampling_rate = audio.frame_rate
            channels = audio.channels
            data = np.array(audio.get_array_of_samples()).reshape((-1, channels))

    elif file_ext == '.mp3':
        audio = AudioSegment.from_file(input_file)
        if sampling_rate is None:
            sampling_rate = audio.frame_rate
        channels = audio.channels
        data = np.array(audio.get_array_of_samples()).reshape((-1, channels))
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    if sampling_rate is None:
        raise ValueError("Sampling rate must be provided for non-audio files")

    # Normalize the data to the range [-1, 1]
    data = data / np.max(np.abs(data))

    # Determine the appropriate data type for the WAV file
    if np.issubdtype(data.dtype, np.integer):
        dtype = 'int16'
    else:
        dtype = 'float32'

    #return wav object together
    return sampling_rate, data.astype(dtype)
    