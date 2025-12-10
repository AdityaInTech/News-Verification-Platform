# 📢 News Verification Platform

An AI-powered system that detects **fake news** using Machine Learning (Logistic Regression + TF-IDF) and provides a confidence score.  
This project includes a trained ML model, Flask backend API, and a responsive frontend interface for real-time news verification.

---

## 🚀 Features

- 🧠 ML-based Fake News Detection  
- ⚡ Real-time prediction with confidence score  
- 🎨 Responsive frontend (HTML, CSS, JS)  
- 🔌 Flask API backend  
- 🗂️ Automatic preprocessing & stemming  
- 🎯 High accuracy (98%+)  
- 📦 Saved model for fast predictions  

---

## 📂 Project Structure

```
📁 News-Verification-Platform
│
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # TF-IDF vectorizer
├── X_test.pkl / Y_test.pkl # Stored test data
│
├── True.csv # Real news dataset
├── Fake.csv # Fake news dataset
│
├── app.py # Flask backend
├── index.html # Frontend UI
├── style.css # Styling
├── script.js # Frontend JS logic
│
└── README.md # Project documentation
```
