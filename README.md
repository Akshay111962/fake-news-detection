# fake-news-detection
# 📰 Fake News Detection System

A Machine Learning web application that detects fake news articles
using Natural Language Processing (NLP) techniques.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-94.64%25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 🔥 Live Demo
[Click here to try the app](#) ← (add Streamlit link after deployment)

## 📊 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 94.64% | 0.95 |
| XGBoost | 95.06% | 0.95 |

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML Libraries:** Scikit-learn, XGBoost
- **NLP:** TF-IDF Vectorization
- **Frontend:** Streamlit
- **Dataset:** WELFake (71,537 articles)

## 📁 Project Structure
```
fake-news-detection/
│
├── app.py                 # Streamlit web app
├── fake_news.ipynb        # Model training notebook
├── logistic_model.pkl     # Trained LR model
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── .gitignore
└── README.md
```
## ⚙️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/Akshay111962/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## 🧠 How It Works

1. User pastes a news article
2. Text is cleaned and preprocessed
3. TF-IDF converts text to numbers
4. Model predicts Fake / Real / Unverified
5. Confidence score is displayed

## 📈 Results

- Trained on **71,537** diverse news articles
- Achieves **94.64% accuracy** on test set
- Precision: **94%** | Recall: **95%** | F1: **95%**

## ⚠️ Limitations

- Best results on political and general news
- Limited performance on science/sports topics
- Trained on English language articles only

## 🔮 Future Improvements

- Add BERT transformer model
- Include diverse datasets (NELA-GT, AllSides)
- Add multilingual support
- Real-time news URL scraping

## 👨‍💻 Author

**Akshay** — [GitHub](https://github.com/Akshay111962)
