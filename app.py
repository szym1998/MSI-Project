from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from flask import send_file
from pydub import AudioSegment
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'datastore'
ALLOWED_EXTENSIONS = {'csv', 'mat', 'txt', 'xlsx', 'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def read_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    return audio.frame_rate, samples

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/set-sampling-frequency/<filename>', methods=['GET', 'POST'])
def set_sampling_frequency(filename):
    if request.method == 'POST':
        sampling_frequency = float(request.form['sampling_frequency'])
        input_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0] + '.wav')
        convert_to_wav(input_file, output_file, sampling_rate=sampling_frequency)
        return redirect(url_for('display_data', filename=output_file))
    else:
        # Set a default sampling frequency for user input
        default_frequency = 44100
        return render_template('set_sampling_frequency.html', filename=filename, default_frequency=default_frequency)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the file to the upload folder
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0] + '.wav')
            
            try:
                convert_to_wav(input_file, output_file)
            except ValueError:
                # Redirect to set_sampling_frequency if ValueError (no sampling frequency) is raised
                return redirect(url_for('set_sampling_frequency', filename=filename))
                
            return redirect(url_for('display_data', filename=output_file))
    return render_template('upload.html')




@app.route('/display-data/<filename>/<float:sampling_frequency>')
def display_data(filename, sampling_frequency):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_ext = os.path.splitext(filename)[1].lower()

    # Read the file using pandas
    if file_ext in {'.csv', '.txt'}:
        data = pd.read_csv(file_path, header=None)
    elif file_ext == '.xlsx':
        data = pd.read_excel(file_path, header=None)
    elif file_ext == '.wav':
        from scipy.io import wavfile
        _, data = wavfile.read(file_path)
    elif file_ext == '.mp3':
        _, data = read_mp3(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Create a time-domain plot using matplotlib
    time = np.arange(0, len(data) / sampling_frequency, 1 / sampling_frequency)
    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(time, data.iloc[:, i], label=f"Column {i+1}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time-domain Plot')
    plt.legend()

    # Save the plot to a BytesIO object and send it as a PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
