from multiprocessing import Pool
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import ploting as plots
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


def get_image_string(plot_function, file_path):
    result = plot_function(file_path)
    return result

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    file_path = request.form.get('file')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)

    time_domain_xlim_str = request.form.get('time_domain_xlim')
    time_domain_xlim = [float(value) if value else None for value in time_domain_xlim_str.split(',')]

    time_domain_ylim_str = request.form.get('time_domain_ylim')
    time_domain_ylim = [float(value) if value else None for value in time_domain_ylim_str.split(',')]

    stft_xlim_str = request.form.get('stft_xlim')
    stft_xlim = [float(value) if value else None for value in stft_xlim_str.split(',')]

    stft_ylim_str = request.form.get('stft_ylim')
    stft_ylim = [float(value) if value else None for value in stft_ylim_str.split(',')]

    stft_cbar_lim_str = request.form.get('stft_cbar_lim')
    stft_cbar_lim = [float(value) if value else None for value in stft_cbar_lim_str.split(',')]

    stft_n_fft = float(request.form.get('stft_n_fft')) if request.form.get('stft_n_fft') else None
    stft_hop_length = float(request.form.get('stft_hop_length')) if request.form.get('stft_hop_length') else None

    fcwt_f0_f1_str = request.form.get('fcwt_f0_f1')
    fcwt_f0_f1 = [float(value) if value else None for value in fcwt_f0_f1_str.split(',')]

    fcwt_xlim_str = request.form.get('fcwt_xlim')
    fcwt_xlim = [float(value) if value else None for value in fcwt_xlim_str.split(',')]

    fcwt_ylim_str = request.form.get('fcwt_ylim')
    fcwt_ylim = [float(value) if value else None for value in fcwt_ylim_str.split(',')]

    fcwt_clim_str = request.form.get('fcwt_clim')
    fcwt_clim = [float(value) if value else None for value in fcwt_clim_str.split(',')]

    fcwt_fn = float(request.form.get('fcwt_fn')) if request.form.get('fcwt_fn') else None
    fcwt_mor_size = float(request.form.get('fcwt_mor_size')) if request.form.get('fcwt_mor_size') else None

    fft_xlim_str = request.form.get('fft_xlim')
    fft_xlim = [float(value) if value else None for value in fft_xlim_str.split(',')]

    fft_ylim_str = request.form.get('fft_ylim')
    fft_ylim = [float(value) if value else None for value in fft_ylim_str.split(',')]



    print(fcwt_f0_f1)

    # Get the list of WAV files
    wav_files = sorted([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.wav')])

    # Create a multiprocessing Pool
    pool = Pool(processes=4)

    # Use pool.apply_async to run the plot methods in parallel, getting a multiprocessing.pool.AsyncResult for each one
    results = [pool.apply_async(get_image_string, args=(plot_function, file_path)) for plot_function in (
        plots.plot_td, plots.plot_stft, plots.plot_fcwt, plots.plot_fft)]

    # Close the pool to prevent any more tasks from being submitted to the pool.
    pool.close()

    # Wait for all the tasks to complete
    pool.join()

    # Combine all results in a single dictionary
    results_dict = {}
    for result in results:
        results_dict.update(result.get())

    # Convert tuples to lists within results_dict
    for key, value in results_dict.items():
        if isinstance(value, tuple):
            results_dict[key] = list(value)

    # Extract the required variables from the results_dict
    image_string_td = results_dict["image_string_td"]
    image_string_stft = results_dict["image_string_stft"]
    image_string_fcwt = results_dict["image_string_fcwt"]
    image_string_fft = results_dict["image_string_fft"]

    xlim_td_var = results_dict["xlim_td"]
    ylim_td_var = results_dict["ylim_td"]
    
    xlim_stft_var = results_dict["xlim_stft"]
    ylim_stft_var = results_dict["ylim_stft"]
    clim_stft_var = results_dict["clim_stft"]
    nfft_stft_var = results_dict["nfft_stft"]
    hop_length_stft_var = results_dict["hop_length_stft"]
    xlim_fcwt_var = results_dict["xlim_fcwt"]
    ylim_fcwt_var = results_dict["ylim_fcwt"]
    clim_fcwt_var = results_dict["clim_fcwt"]
    f1_fcwt_var = results_dict["f1_fcwt"]
    xlim_fft_var = results_dict["xlim_fft"]
    ylim_fft_var = results_dict["ylim_fft"]


    # Pass the results dictionary to the template
    return render_template('analysis.html', files=wav_files, 
                        image_string_time_domain=image_string_td, 
                        image_string_stft=image_string_stft,
                        image_string_fcwt=image_string_fcwt,
                        image_string_fft=image_string_fft,
                        xlim_td=xlim_td_var,
                        
                        ylim_td=ylim_td_var,
                        xlim_stft=xlim_stft_var,
                        ylim_stft=ylim_stft_var,
                        clim_stft=clim_stft_var,
                        nfft_stft=nfft_stft_var,
                        hop_length_stft=hop_length_stft_var,
                        xlim_fcwt=xlim_fcwt_var,
                        ylim_fcwt=ylim_fcwt_var,
                        clim_fcwt=clim_fcwt_var,
                        f1_fcwt=f1_fcwt_var,
                        xlim_fft=xlim_fft_var,
                        ylim_fft=ylim_fft_var)
                           

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000)
