from flask import Flask, render_template, request, make_response, jsonify
import spacy
from beams_clean import Beams
from prettytable import *
from flask_dropzone import Dropzone


app = Flask(__name__)
dropzone = Dropzone(app)

nlp = spacy.load('../IMaSC')
print("here")

@app.route('/')
def pee():
    return render_template('api_home.html')


@app.route('/teapot', methods=['GET', 'POST'])
def poop():
    if request.method == 'POST':
        print('HERE')
        f = request.files.get('file')
        f.save(os.path.join('/', f.filename))
    user_input = request.args.get("user_string")
    to_feed = []
    to_feed.append(user_input)
    b = Beams(model_path = "../IMaSC", data_list = to_feed)
    # guesses = ""
    # doc = nlp(user_input)
    # a = set()
    # for ent in doc.ents:
    #     a.add((ent.text, ent.label_))
    a = set()
    b.perform()
    a = b.get_scores()
    if len(a) == 0:
        return "IMaSC found no results."
    result = list(a)
    print(result)
    rows = []
    table = PrettyTable(["Token","Tag","Confidence Score"])
    for line in result:
        print(list(line))
        rows.append(list(line))
    print(rows)
    return render_template("info.html", title="Confidence Scores", rows=rows)


@app.before_request
def before():
    print("This is executed BEFORE each request.")
