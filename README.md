# Motivation
This project features pyhon scripts that run ETL and NLP pipelines in order to create a machine learning model that categorises message into categories for use in natural disasters. This model is then incorporated into a web app that can be used to categorise any message the user inputs.

This project forms part of Udacity's Data Scientist nanodegree program.


# Libraries
The python scripts require the following libraries:
* numpy
* pandas
* sqlite3
* sklearn
* nltk
* joblib
* json
* plotly
* flask

The HTML code uses the following libraries:
* boostrap
* jquery
* plotly


# How to run the project
## ETL process
The process_data.py file runs the ETL process that merges and cleans two input datasets.

This file can best be run from a Linux or Mac terminal, and it takes three arguments: filepath to messages dataset, filepath to categories dataset, and filepath for the database that will be created.

Example:
  python process_data.py messages.csv categories.csv CategorisedMessages.db

This creates CategorisedMessages.db database file, with the final data stored as table Messages.

## NLP pipeline and ML model
Once the data has been cleaned, the NLP and ML processes can be run.

This is the train_classifier.py file, and it takes three arguments when run in the terminal: filepath to the database and a filepath for the output model.

Example:
  python train_classifier.py ../data/CategorisedMessages.db classifier.pkl

## Running the web app
Once the model has been created, the web app can be run by running the run.py file in terminal:
  python run.py

# Full list of files
* **app/**
  * **templates/**
    * **master.html**: html file responsible for the majority of the web app.
    * **go.html**: extension to master.html that displays the categories of user's input message.
  * **run.py**: python file that runs the web app.
* **data/**
  * **categories.csv**: dataset provided by Figure Eight with message categories
  * **CategorisedMessages.db**: database file that stores cleaned and processed data (output of process_data.py file)
  * **messages.csv**: dataset provided by Figure Eight with raw messages.
  * **process_data.py**: python script that merges and cleans the above two datasets
* **models/**
  * **train_classifier.py**: python script that takes the database output of process_data.py and runs an NLP and ML processes to output a model that can categorise messages.

Note that while the output of process_data.py is provided here (CategorisedMessages.db), the output of train_classifier.py is not, as the resulting pickle file is too large to be stored on GitHub.

# Acknowledgements
Thanks to Udacity, who provided the majority of the web app code.

Thanks to Figure Eight, who provided the messages and categories datasets through Udacity.
