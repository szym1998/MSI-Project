import numpy as np
from scipy.io import wavfile

def wav_to_csv(input_file, output_file):
    # Read the WAV file
    sampling_rate, data = wavfile.read(input_file)
    print(sampling_rate)

    # Convert the data to a 2D array with one column for each channel
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Write the data to a CSV file
    np.savetxt(output_file, data, delimiter=',')






input_file = 'files/file_example_MP3_700KB.wav'
output_file = 'files/file_example_MP3_700KB.csv'

wav_to_csv(input_file, output_file)
