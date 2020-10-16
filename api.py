from flask import Flask

app = Flask(__name__)


@app.route('/special/<int:number>/')
def incrementer(number):
    return "Incremented number is " + str(number+1)


@app.route('/special/<string:name>/')
def hello(name):
    return "Hello " + name

@app.route('/teapot/')
def teapot():
    return "Would you like some tea?", 418

@app.before_request
def before():
    print("This is executed BEFORE each request.")

app.run()
