from model import predict
from flask import request
import flask
import os

# Initialize the app
app = flask.Flask(__name__)


@app.route('/')
def index():

    # Displays the shown string above the user entered text
    header_text = "Text:"

    # Displays the show string above the model determined sentiment
    header_pred = "Classification:"

    print(request.args)

    # Contains a dictionary containing the parsed contents of the query string
    if(request.args):

        # Passes contents of query string to the prediction function contained in model.py
        x_input, prediction = predict(request.args['text_in'])
        print(prediction)

        # Indexes the returned dictionary for the sentiment probability
        return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_text=header_text, header_pred=header_pred)

    # If the parsed query string does not contain anything then return index page
    else:
        return flask.render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
