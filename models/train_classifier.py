import sys
import joblib

import numpy as np
import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords'])


def load_data(db_path):
    '''Input: filepath to database.

    Function loads data from SQLite database, and returns as Pandas dataframe.
    '''
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('select * from Messages', conn)
    df = df.set_index('id', drop=True)
    return df

def get_wordnet_pos(treebank_tag):
    '''Takes in part of speech result from pos_tag(), and converts for use
    with WordNetLemmatizer().lemmatize().
    '''
    if treebank_tag.startswith('J'):     #adjective
        return 'a'
    elif treebank_tag.startswith('V'):   #verb
        return 'v'
    elif treebank_tag.startswith('N'):   #noun
        return 'n'
    elif treebank_tag.startswith('R'):   #adverb
        return 'r'
    else:
        return 'n'        #lemmatize() defaults to 'n' if unknown

def tokenize(txt):
    '''Input: string.
    Output: clean token list.

    1. Normalizes text (all lower, no punct).
    2. Tokenizes text.
    3. Removes stop words.
    4. Tags pos.
    5. Lemmatizes text.
    '''
    txt = word_tokenize(txt)
    txt = [w.lower() for w in txt if w.isalnum()]
    txt = [w for w in txt if w not in stopwords.words('english')]
    txt = pos_tag(txt)
    txt = [WordNetLemmatizer().lemmatize(tup[0], pos=get_wordnet_pos(tup[1]))
        for tup in txt]

    return txt

def split_and_train(df):
    '''Input: dataframe.
    Output: trained model.

    Function splits df into training and testing sets, runs an NLP and ML
    pipeline, and optimises hyperparameters with GridSearchCV.
    '''

    X = df['message']
    y = df.drop(['message', 'original', 'genre'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.33, random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier(n_jobs=-2)
                    , n_jobs=-2))
    ])

    parameters = {
        'vect__binary': [True, False],
        'tfidf__smooth_idf': [True, False],
        'model__estimator__n_estimators': [30, 50, 100, 300],
        'model__estimator__max_depth': [None, 5, 10]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=1)

    cv.fit(X_train, y_train)

    print('The best hyperparameters are:')
    print(cv.best_params_)

    '''
    The best hyperparameters are:
    {'model__estimator__max_depth': None,
    'model__estimator__n_estimators': 300,
    'tfidf__smooth_idf': False,
    'vect__binary': False}
    '''

    y_pred = cv.predict(X_test)

    print('Printing classification_report for each target:')
    for x in range(len(y.columns)):
        print(classification_report(y_test.iloc[:,x], y_pred[:,x]))

    return cv

def output_pickle(model, model_path):
    '''Input: trained model, and filepath for output pickle.

    This function takes the input model, and exports it as a pickle file.
    '''

    joblib.dump(model.best_estimator_, model_path, compress=1)

if __name__ == '__main__':
    db_path = sys.argv[1]
    model_path = sys.argv[2]

    df = load_data(db_path)
    model = split_and_train(df)
    output_pickle(model, model_path)
