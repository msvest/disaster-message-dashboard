import json
import plotly
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
import sqlite3



app = Flask(__name__)

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

# load data
conn = sqlite3.connect('../data/CategorisedMessages.db')
df = pd.read_sql('select * from Messages', conn)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    #proportion of messages that are non-English
    bar_labels = ['English', 'Non-English']
    bar_values = [df['message'].count()-df['original'].count(), df['original'].count()]

    bar_graph = [Bar(
        x = bar_labels,
        y = bar_values
    )]

    layout_one = dict(
        title = 'Distribution of Message Language',
        xaxis = dict(title = 'Language'),
        yaxis = dict(title = 'Count')
    )

    #doughnut chart
    counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    pie_labels = counts.index.tolist()
    pie_values = counts.values.tolist()

    pie_graph = [Pie(
        labels=pie_labels,
        values=pie_values
    )]

    layout_two = dict(
        title = 'Distribution of Message Category'
    )

    figures = []
    figures.append(dict(data=bar_graph, layout=layout_one))
    figures.append(dict(data=pie_graph, layout=layout_two))

    # encode plotly graphs in JSON
    ids = ['graph_one', 'graph_two']
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query]).tolist()[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
