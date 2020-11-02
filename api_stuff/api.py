from flask import Flask, render_template, request, make_response, jsonify
import spacy

app = Flask(__name__)

nlp = spacy.load('../IMaSC')

@app.route('/')
def pee():
    return render_template('api_home.html')

@app.route('/teapot', methods=['GET'])
def poop():
    user_input = request.args.get("user_string")
    guesses = ""
    doc = nlp(user_input)
    a = set()
    for ent in doc.ents:
        a.add((ent.text, ent.label_))
    result = list(a)
    if len(a) == 0:
        return "IMaSC found no results."
    return jsonify(results = result )

@app.before_request
def before():
    print("This is executed BEFORE each request.")

app.run()
