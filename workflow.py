import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from prefect import task,flow

@task
def load_data(file_path):
    return pd.read_csv(file_path)


@task
def split_inputs_output(data, inputs, output):
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X_train, X_test, y_train, y_test):
    vocab = CountVectorizer()
    X_train_bow = vocab.fit_transform(X_train['Review'])
    X_test_bow = vocab.transform(X_test['Review'])
    return X_train_bow, X_test_bow, y_train, y_test

@task
def train_model(X_train_scaled, y_train, hyperparameters):
    clf = DecisionTreeClassifier(**hyperparameters)
    clf.fit(X_train_scaled, y_train)
    return clf

@task
def evaluate_model(model, X_train_bow, y_train, X_test_bow, y_test):
    y_train_pred = model.predict(X_train_bow)
    y_test_pred = model.predict(X_test_bow)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score


@flow(name="Decision Tree Training Flow")
def workflow():
    DATA_PATH = "output.csv"
    INPUTS = ['Review']
    OUTPUT = 'sentiment'
    HYPERPARAMETERS = {'max_depth': 10}
    
    # Load data
    sentiment = load_data(DATA_PATH)

    # Identify Inputs and Output
    X, y = split_inputs_output(sentiment, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    X_train=X_train.to_frame()
    X_test=X_test.to_frame()

    # Preprocess the data
    X_train_bow, X_test_bow, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    # Build a model
    model = train_model(X_train_bow, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train_bow, y_train, X_test_bow, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    

if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="* * * * *"
    )