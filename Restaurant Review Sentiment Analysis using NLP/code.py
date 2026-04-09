import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (run once)
nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv(r"C:\Users\VIRBHADRA\Downloads\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

# Cleaning text
ps = PorterStemmer()
corpus = []

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Feature extraction (Better than CountVectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Best model for NLP
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

bias1 = model.score(X_train,y_train)
print(bias1)

variance1 = model.score(X_test,y_test)
print(variance1)

import pickle

# save model
pickle.dump(model, open('model.pkl', 'wb'))

# save vectorizer
pickle.dump(cv, open('vectorizer.pkl', 'wb'))