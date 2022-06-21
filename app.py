from flask import Flask, request, render_template
from main import *

app = Flask(__name__)
@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        user = request.form['moviereview']
        test = custom_input(user, cv, classifier)
        label = ""
        if (test == 0):
            label = "negative"
            return render_template("index.html", data=label)
        else:
            label = "positive"
            return render_template("index.html", data=label)
    else:
        label = ""
        return render_template("index.html", data=label)

if __name__ == '__main__':
    n = int(input("1.Train the model\n2.Test custom review\n"))
    if n == 1:
        main()
    else:
        classifier, cv = load()
        app.run()


