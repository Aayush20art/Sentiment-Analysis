# 🎭 VibeCheck AI – Emotion Sentiment Analysis

A **Machine Learning powered Emotion Detection Web App** that analyzes text and predicts the human emotion behind it in real time.

Built using **Natural Language Processing (NLP)**, **Scikit-learn**, and **Streamlit**, this project demonstrates a complete ML pipeline from text preprocessing to model prediction and visualization.

---

## 🌐 Live Demo

🚀 Try the app here:  
[Sentiment Analysis Web App](https://sentiment-analysis-jlyvravbfgnfhf4mdep6i6.streamlit.app/)

---

## 📌 Features

- ✨ Real-time Emotion Detection from Text  
- ✨ Clean interactive Streamlit UI  
- ✨ Supports **6 emotions classification**:  
  - 😄 Joy  
  - 😢 Sadness  
  - 😠 Anger  
  - 😨 Fear  
  - ❤️ Love  
  - 😲 Surprise  
- ✨ Multiple ML models comparison  
- ✨ NLP preprocessing pipeline visualization  
- ✨ Model performance analytics with charts  

---

## 🧠 Machine Learning Models

| Model                   | Feature Extraction | Purpose                |
|--------------------------|-------------------|------------------------|
| Multinomial Naive Bayes  | Bag of Words      | Fast baseline          |
| Multinomial Naive Bayes  | TF-IDF            | Improved baseline      |
| Logistic Regression      | TF-IDF            | ⭐ Best performing model |

---

## ⚙️ NLP Pipeline

The text passes through several preprocessing steps:

1. Lowercasing  
2. Removing punctuation  
3. Removing numbers  
4. Removing non-ASCII characters  
5. Removing stopwords (NLTK)  
6. Bag of Words vectorization  
7. TF-IDF transformation  
8. Model prediction  

---

## 🖥️ Tech Stack

- 🐍 Python  
- 📊 Scikit-learn  
- 🌿 NLTK  
- 🐼 Pandas & NumPy  
- 📈 Matplotlib & Seaborn  
- ⚡ Streamlit  

---

## 📂 Project Structure

sentiment-analysis/
│
├── app.py                # Streamlit application
├── train.txt             # Emotion dataset
├── model.pkl             # Saved ML model
├── vectorizer.pkl        # Saved vectorizer
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

Code

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
Install dependencies:

bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
📊 Example Prediction
Input Text:

Code
"I feel so happy today! Everything is going great."
Prediction:

Emotion: Joy 😄

Confidence: 92%

🎯 Future Improvements
Deep Learning models (LSTM / BERT)

More emotions classification

API integration

Speech emotion detection

Larger dataset

👨‍💻 Author
Aayush Sharma  
Aspiring Data Scientist | Machine Learning Enthusiast
