# import libraries
import sys

import re
import pickle
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


# load data from database
def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    table_name='DisasterResponse'
    con='sqlite:///./data/DisasterResponse.db'
    df = pd.read_sql_table(table_name=table_name, con=con)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y

def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens   


def build_model():
    """
    Build Pipeline function with GridSearchCV for Hyperparameter tuning
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [100,150],
        'clf__estimator__min_samples_split': [2,3],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=5)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        model  -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        print(y_test.columns[i], ':')
        print(classification_report(y_test.values[:,i], 
                                    y_pred[:,i],
                                    target_names=[f'class {i}' for i in range(y_test.iloc[:,i].nunique())]),
              '...................................................')

def save_model(model, model_filepath):
    """
    Saves a machine learning model
    
    This function saves trained ML model as Pickle file, to be re-loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit-learn Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()