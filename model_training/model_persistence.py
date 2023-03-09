import pickle

def save_model(model, file_path):
    """ Pickles the model to be used later. """
    pickle.dump(model, open(file_path, 'wb'))

def load_model(file_path):
    """ Load the model from a pickle file and return it. """
    model = pickle.load(open(file_path, 'rb'))
    return model