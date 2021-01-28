from flask import Flask, render_template, request, make_response, jsonify
import spacy
from beams_clean import Beams
from prettytable import *
from flask_dropzone import Dropzone
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from tika import parser
import os


app = Flask(__name__)

nlp = spacy.load('../IMaSC')
dropzone = Dropzone(app)

@app.route('/')
def pee():
    return render_template('api_home.html')


@app.route('/teapot', methods=['GET', 'POST'])
def poop():
    if request.method == 'POST':
        f = request.files['file']
        #f = request.args.get('file')
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        f.save(os.path.join(APP_ROOT, "myspecialfile"))
        print("HERE1")
        file_data = parser.from_file(os.path.join(APP_ROOT, "myspecialfile"))
        print('HERE2')
        text = file_data['content']
    else:
        text = request.args.get("user_string")
    to_feed = []
    to_feed.append(text)
    b = Beams(model_path = "../IMaSC", data_list = to_feed)
    a = set()
    b.perform()
    a = b.get_scores()
    if len(a) == 0:
        return "IMaSC found no results."
    result = list(a)
    rows = []
    table = PrettyTable(["Token","Tag","Confidence Score"])
    coverage_thresh = 0.9
    coverage_count = 0.0
    for line in result:
        if float(line[2]) >= coverage_thresh:
            coverage_count += 1
        print(list(line))
        new_list = [line[0], line[1], round(float(line[2]), 3)]
        rows.append(new_list)
    print("TOTAL: ", len(result))
    print("PAST THRESH: ", coverage_count)
    print("COVERAGE: ", coverage_count / len(result))
    cov = coverage_count / len(result)
    return render_template("info.html", title="Confidence Scores", rows=rows, cov=cov)



@app.before_request
def before():
    print("This is executed BEFORE each request.")

app.run()
