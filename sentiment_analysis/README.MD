# Sentiment Analysis using Naive Bayes

## Overview
This project performs sentiment analysis on text data using a Naive Bayes classifier. The dataset consists of sample text reviews labeled as positive (1) or negative (0). The text data is preprocessed, transformed into numerical features using TF-IDF, and classified using Multinomial Naive Bayes.

## Requirements
To run this project, install the following dependencies:
```bash
pip install pandas numpy nltk seaborn scikit-learn matplotlib
```
Additionally, download necessary NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Dataset
The dataset is a dictionary with text samples and corresponding sentiment labels:
```python
data = {'text': [
    "I love this product! It's amazing!",
    "This is the worst experience ever. Never buying again!",
    "The quality is okay, not the best but decent.",
    "Absolutely fantastic! Highly recommend.",
    "Terrible service, very disappointed.",
    "Had an average experience, nothing special.",
    "Great value for money, satisfied with the purchase.",
    "I hate this, it broke after one use!"
],
    'sentiment': [1, 0, 1, 1, 0, 1, 1, 0]
}
```

## Text Preprocessing
1. Convert text to lowercase.
2. Remove special characters and numbers.
3. Tokenize words.
4. Remove stopwords.

## Model Training
1. Convert text to numerical features using `TfidfVectorizer`.
2. Split data into training (80%) and testing (20%) sets.
3. Train a Naive Bayes classifier (`MultinomialNB`).

## Evaluation
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization

## Running the Project
Run the Python script to execute sentiment analysis:
```bash
python sentiment_analysis.py
```

## Output Example
```
Accuracy: 0.75

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.50      0.67         2
           1       0.67      1.00      0.80         2

    accuracy                           0.75         4
   macro avg       0.83      0.75      0.73         4
weighted avg       0.83      0.75      0.73         4
```

## Visualization
A confusion matrix is plotted to show model performance.

## License
This project is open-source and available for modification and distribution.

