🎭 VibeCheck AI – Emotion Sentiment Analysis

A Machine Learning powered Emotion Detection Web App that analyzes text and predicts the human emotion behind it in real time.

Built using Natural Language Processing (NLP), Scikit-learn, and Streamlit, this project demonstrates a complete ML pipeline from text preprocessing to model prediction and visualization.

🌐 Live Demo

🚀 Try the app here:
https://sentiment-analysis-jlyvravbfgnfhf4mdep6i6.streamlit.app/

📌 Features

✨ Real-time Emotion Detection from Text
✨ Clean interactive Streamlit UI
✨ Supports 6 emotions classification

😄 Joy
😢 Sadness
😠 Anger
😨 Fear
❤️ Love
😲 Surprise

✨ Multiple ML models comparison
✨ NLP preprocessing pipeline visualization
✨ Model performance analytics with charts

🧠 Machine Learning Models

The project compares different ML algorithms:

Model	Feature Extraction	Purpose
Multinomial Naive Bayes	Bag of Words	Fast baseline
Multinomial Naive Bayes	TF-IDF	Improved baseline
Logistic Regression	TF-IDF	⭐ Best performing model
⚙️ NLP Pipeline

The text passes through several preprocessing steps:

Lowercasing
Removing punctuation
Removing numbers
Removing non-ASCII characters
Removing stopwords (NLTK)
Bag of Words vectorization
TF-IDF transformation
Model prediction
🖥️ Tech Stack
🐍 Python
📊 Scikit-learn
🌿 NLTK
🐼 Pandas & NumPy
📈 Matplotlib & Seaborn
⚡ Streamlit
📂 Project Structure
sentiment-analysis/
│
├── app.py                # Streamlit application
├── train.txt             # Emotion dataset
├── model.pkl             # Saved ML model
├── vectorizer.pkl        # Saved vectorizer
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
🚀 Installation

Clone the repository:

git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py
📊 Example Prediction

Input Text:

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

⭐ Support

If you like this project, consider starring the repository ⭐
It helps others discover it!
