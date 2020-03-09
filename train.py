from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import numpy as np

categories = ['sci.med', 'sci.electronics', 'rec.autos']

train = fetch_20newsgroups(subset='train', categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)

from joblib import dump
dump(model, './Predictapp/static/chatgroup.model')
dump(train, './Predictapp/static/train.model')

print(round((model.score(train.data, train.target)*100),2),'%')
