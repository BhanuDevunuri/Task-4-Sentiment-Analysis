# Task 4: Sentiment Analysis

## 📌 Overview
This project performs **Sentiment Analysis on textual data** using **Natural Language Processing (NLP)** techniques.  
It uses a **Naive Bayes classifier** to classify text into **positive or negative sentiments**.

---

## 📌 Features
✔ **Cleans & Preprocesses Text Data** (Stopwords Removal, Tokenization)  
✔ **Converts Text into Numerical Features** using **TF-IDF**  
✔ **Trains a Naive Bayes Classification Model**  
✔ **Evaluates Model Performance** (Accuracy, Confusion Matrix, Classification Report)  
✔ **Visualizes Confusion Matrix using Seaborn**  

---

## 📌 Steps Performed

1️⃣ **Text Preprocessing**
   - Lowercased text & removed special characters.
   - Tokenized words and removed stopwords.

2️⃣ **Feature Engineering**
   - Converted text to **TF-IDF vectors**.

3️⃣ **Model Training**
   - Used **Naive Bayes (MultinomialNB)** for sentiment classification.

4️⃣ **Model Evaluation**
   - Computed **accuracy score**.
   - Generated a **confusion matrix heatmap**.

---

## 📌 How to Run the Code

### **1️⃣ Install Required Libraries**
Ensure you have Python installed. Install dependencies using:

```bash
pip install pandas numpy nltk seaborn scikit-learn matplotlib
