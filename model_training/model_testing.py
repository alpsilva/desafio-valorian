from sklearn.metrics import precision_recall_fscore_support

def test_model(model, X_test, y_test):
    """
    Receives a model, along with a training set.
    Returns the average accuracy, recall and f1.
    """
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred)
    return precision, recall, f1

