import flask
from flask import request
import os
from werkzeug.utils import secure_filename

import generate as gen

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png'])

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def home():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        gen.generate(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    response = flask.jsonify({'some': '<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/return-files/')
def home2():
    response = flask.jsonify({'some': '<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>'})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return flask.send_file('./uploads/enhanced.png', attachment_filename='enhanced.png')


app.run()