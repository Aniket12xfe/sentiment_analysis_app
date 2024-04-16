from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_analysis_model.joblib')


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Get the text input from the user
    text = request.form['text']

    # Use the trained model to predict the sentiment
    sentiment = model.predict([text])[0]

    # Return the result as JSON
    return render_template('result.html', text=text, sentiment=sentiment)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
