import numpy as np
import pandas as pd
import re
import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  

from tqdm import tqdm
tqdm.pandas()

import os
import pickle
# news_data['content'] = news_data['content'].progress_apply(stemming)

import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Check if model and vectorizer are already saved
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

else : 

      true_data = pd.read_csv('True.csv')
      fake_data = pd.read_csv('Fake.csv')

      true_data['label']=1
      fake_data['label']=0

      news_data = pd.concat([true_data,fake_data],axis=0)

      news_data['content'] = news_data['title']
      # +' '+news_data['text']

      news_data['content'] = news_data['content'].progress_apply(stemming)
      
      X = news_data['content'].values
      Y = news_data['label'].values

      vectorizer = TfidfVectorizer()
      vectorizer.fit(X)

      X = vectorizer.transform(X)

      X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)

      model = LogisticRegression()

      model.fit(X_train,Y_train)

      with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
      with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

      # Save test set so accuracy to access later
      with open("X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
      with open("Y_test.pkl", "wb") as f:
        pickle.dump(Y_test, f)

       # Calculate and print overall accuracy after training
      Y_pred = model.predict(X_test)
      accuracy = accuracy_score(Y_test, Y_pred)
      print(f"Model Accuracy on test set: {accuracy*100:.2f}%")


news = input("Enter News")

numeric_news = vectorizer.transform([news])

prediction = model.predict(numeric_news)

prediction_proba = model.predict_proba(numeric_news)

if prediction[0] == 0:
    print(f"Fake News (Confidence: {prediction_proba[0][0]*100:.2f}%)")  # generated: show confidence for fake
else:
    print(f"Real News (Confidence: {prediction_proba[0][1]*100:.2f}%)")  # generated: show confidence for real