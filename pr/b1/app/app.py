from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import classifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/upload/'
app.config['SECRET_KEY'] = 'd3Y5d5nJkU6CdwY'
if os.path.exists(app.config['UPLOAD_FOLDER']):
    print("directory exists")
else:
    os.makedirs(app.config['UPLOAD_FOLDER'])
    print("directory created")


@app.route("/", methods=["GET", "POST"])
def home():
    algorithms = {'Neural Network': '92.26 %', 'Support Vector Classifier': '89 %'}
    result, accuracy, name, sdk, size = '', '', '', '', ''
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.apk'):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if request.form['algorithm'] == 'Neural Network':
                accuracy = algorithms['Neural Network']
                result, name, sdk, size = classifier.classify(os.path.join(app.config['UPLOAD_FOLDER'], filename), 0)
            elif request.form['algorithm'] == 'Support Vector Classifier':
                accuracy = algorithms['Support Vector Classifier']
                result, name, sdk, size = classifier.classify(os.path.join(app.config['UPLOAD_FOLDER'], filename), 1)
    return render_template("index.html", result=result, algorithms=algorithms.keys(), accuracy=accuracy, name=name,
                           sdk=sdk, size=size)


if __name__ == "__main__":  # on running python app.py
    app.run(debug=True)  # run the flask app
