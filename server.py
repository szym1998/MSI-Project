from flask import Flask, render_template, request, redirect, url_for, flash
import os
import ploting as pl
import base64
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'mat', 'csv'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def analysis_up():
    if request.method == 'POST':
        # Check if a file was posted
        if 'file' not in request.files:
            
            return redirect(request.url)
        
        file = request.files['file']

        # If user does not select file, the browser might
        # submit an empty part without filename
        if file.filename == '':
            
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('analysis.html', files=files)


@app.route('/', methods=['POST'])
def submit_analysis():
    selected_file = request.form.get('file')
    selected_analyses = request.form.getlist('analysis')
    
    if not selected_file or not selected_analyses:
        # Return an error message if no file or analyses were selected
        return render_template('error.html', message="Please select a file and at least one analysis method.")
    
    # Generate the plots here based on the selected file and analyses...
    
    return "Analysis submitted for file: " + selected_file + ", Analyses: " + ", ".join(selected_analyses)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def receive_file():
    if 'file' not in request.files:
        return render_template('error.html', message="No file part in the request.")
    file = request.files['file']
    
    if file.filename == '':
        return render_template('error.html', message="No file selected for uploading.")
    
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return redirect(url_for('analysis'))




from multiprocessing import Pool

def get_image_string(plot_function, file_path):
    result = plot_function(file_path)
    image_string = base64.b64encode(result[0].getvalue()).decode('utf-8')  # Get the first returned value (buf) only
    return image_string


@app.route('/analysis', methods=['POST'])
def analysis():
    file_path = request.form.get('file')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
    method = request.form.get('method')

    print(method)

    # Get the list of WAV files
    wav_files = sorted([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.wav')])

    # Create a multiprocessing Pool
    pool = Pool(processes=4) # You can set this to the number of cores you want to use.

    # Use pool.apply_async to run the plot methods in parallel, getting a multiprocessing.pool.AsyncResult for each one
    results = [pool.apply_async(get_image_string, args=(plot_function, file_path)) for plot_function in (pl.plot_time_domain, pl.plot_frequency_domain, pl.plot_fcwt, pl.plot_fft)]

    # Close the pool to prevent any more tasks from being submitted to the pool.
    pool.close()

    # Wait for all the tasks to complete
    pool.join()

    # Get the results
    image_string_time_domain, image_string_stft, image_string_fcwt, image_string_fft = [result.get() for result in results]

    return render_template('analysis.html', files=wav_files, image_string_time_domain=image_string_time_domain, image_string_stft=image_string_stft, image_string_fcwt=image_string_fcwt, image_string_fft=image_string_fft)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000)
