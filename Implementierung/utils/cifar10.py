from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def load_and_preprocess(preprocess_fct=None, dtype_X=None):
    # Load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # One-Hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # If dtype_X is given convert to dtype_X
    if dtype_X is not None:
        X_train, X_test = X_train.astype(dtype_X), X_test.astype(dtype_X)

    if preprocess_fct is not None:
        # Preprocess train and test set as defined by the architecture
        X_train = preprocess_fct(X_train)
        X_test = preprocess_fct(X_test)

    # Define class labels
    class_labels = [
        "Flugzeug", "Automobil",
        "Vogel", "Katze",
        "Hirsch", "Hund",
        "Frosch", "Pferd",
        "Schiff", "Lastkraftwagen",
    ]

    return (X_train, y_train), (X_test, y_test), class_labels
