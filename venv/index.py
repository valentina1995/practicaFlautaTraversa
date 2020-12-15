from flask import Flask, request, render_template
import os
import sys
import librosa
import pandas as pd
import numpy as np
import pretty_midi
import module
from werkzeug.utils import secure_filename
TEMPLATE_DIR = os.path.abspath('templates')


UPLOAD_FOLDER = './audio'
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder='C:/Users/Valen/Desktop/myproject/venv/static', static_url_path="/static")
app.secret_key = 'super secret'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')


@app.route('/results')
def show_results():
    filename = request.args.get('filename', None)
    filename_Mid = request.args.get('filenameMid', None)
    y, sr = librosa.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    midi_data = pretty_midi.PrettyMIDI(os.path.join(app.config['UPLOAD_FOLDER'], filename_Mid))
    #Extraemos el hop_length de acuerdo a la velocidad en bmp del ejercicio
    hop_length, bmp = module.extractHopLenght(y,sr)    
    #Segmentamos la señal
    sep, div = module.divSignal(y,hop_length)
    #Creamos dataframes con cada archivo
    df_audio = module.get_df_audio(sep, div, bmp, sr)
    df_mid = module.get_df_mid(midi_data)
    grade, wr_note, wr_time, overall = module.note_calification(df_audio, df_mid)
    print(grade)
    
    return render_template('results.html', grade = grade.to_dict(orient='records'), wr_note = wr_note, wr_time = wr_time, overall = overall)


@app.route("/performance", methods=['GET','POST'])
def uploader_audio():
#Falta que cargue el mid
    if request.method == 'POST':
        if 'audio' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if 'midi' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['audio']
        midfile = request.files['midi']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if (file and module.allowed_file(file.filename))and (midfile and module.allowed_file(midfile.filename)):
            filename = secure_filename(file.filename)
            filenameMid = secure_filename(midfile.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            midfile.save(os.path.join(app.config['UPLOAD_FOLDER'], filenameMid))
            flash("Archivos subidos exitosamente")
            return redirect(url_for('show_results',filename = filename, filenameMid = filenameMid ))
    return render_template('perfor.html')      
    
    


if __name__=='__main__':
    app.run(debug = True)