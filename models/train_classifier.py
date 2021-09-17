import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    Y = df.iloc[:, 5:].values
    category_names = df.columns[5:]
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> Pipeline:
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names) -> None:
    report = (classification_report(Y_test, model.predict(X_test), output_dict=True))
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-4, :]
    report_df['categories'] = category_names
    print(report_df)


def save_model(model, model_filepath) -> None:
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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