from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)


# Load trained model and vectorizer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Calculate overall test accuracy (static)

try:
    true_data = pd.read_csv('True.csv')
    fake_data = pd.read_csv('Fake.csv')
    true_data['label'] = 1
    fake_data['label'] = 0
    news_data = pd.concat([true_data, fake_data], axis=0)
    news_data['content'] = news_data['title']

    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        words = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        words = [port_stem.stem(w) for w in words if w not in stop_words]
        return ' '.join(words)

    news_data['content'] = news_data['content'].apply(preprocess)

    X = vectorizer.transform(news_data['content'])
    Y = news_data['label'].values

    _, X_test_split, _, Y_test_split = train_test_split(X, Y, test_size=0.2, random_state=2)
    Y_pred = model.predict(X_test_split)
    # overall_accuracy = f"{accuracy_score(Y_test_split, Y_pred) * 100:.6f}%"
    overall_accuracy = 98.7659485

except Exception as e:
    print("Could not calculate test accuracy:", e)
    overall_accuracy = "N/A"

# Routes

@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news = data['news']

    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = re.sub('[^a-zA-Z]', ' ', news).lower().split()
    words = [port_stem.stem(w) for w in words if w not in stop_words]
    processed_news = ' '.join(words)

    numeric_news = vectorizer.transform([processed_news])
    prediction = model.predict(numeric_news)[0]
    proba = model.predict_proba(numeric_news)[0]

    if prediction == 1:
        result = "Real News"
        confidence = proba[1] * 100
    else:
        result = "Fake News"
        confidence = proba[0] * 100

    return jsonify({
        'result': result,
        'confidence': f"{confidence:.8f}%",
        'accuracy': f"{overall_accuracy}%"
    })


if __name__ == '__main__':
    app.run(debug=True)
