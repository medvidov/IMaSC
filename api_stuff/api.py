from flask import Flask, render_template, request
import spacy

app = Flask(__name__)

#nlp = spacy.load('/Users/shaya/Desktop/Git/medvidov/IMaSC/IMaSC')
nlp = spacy.load('../IMaSC')

@app.route('/special/<int:number>/')
def incrementer(number):
    return "Incremented number is " + str(number+1)


@app.route('/special/<string:name>/')
def hello(name):
    return "Hello " + name

@app.route('/')
def pee():
    return render_template('api_home.html')

@app.route('/teapot', methods=['GET'])
def poop():
    user_input = request.args.get("user_string")
    guesses = ""
    doc = nlp(user_input)
    for ent in doc.ents:
        guesses += ent.text + ": " +  ent.label_ + ", \n"
    return guesses

@app.before_request
def before():
    print("This is executed BEFORE each request.")

app.run()
