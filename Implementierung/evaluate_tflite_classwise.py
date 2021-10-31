#!/usr/bin/env python3

import sys
import json
from pprint import pprint

from configs.architectures import architectures
from utils import cifar10, tflite, metrics


def evaluate_tflite(model_path, samples, labels, class_labels=None):
    # Get predictions
    predictions = tflite.predict(model_path, samples)

    # Evaluate classwise accuracy on the predicted labels
    return metrics.top_k_classwise_accuracy(labels, predictions, 1, class_labels)


if __name__ == "__main__":

    ### Extract command line arguments
    model_key = sys.argv[1]
    tflite_path = sys.argv[2]
    info_path = sys.argv[3] if len(sys.argv) > 3 else None


    ### Load and preprocess data
    preprocess_input = architectures[model_key]["preprocessing_fct"]
    (X_train, y_train), (X_test, y_test), class_labels = cifar10.load_and_preprocess(preprocess_input)


    ### Evaluate
    # Evaluate tflite file on train and test set
    train_results = evaluate_tflite(tflite_path, X_train, y_train, class_labels)
    test_results = evaluate_tflite(tflite_path, X_test, y_test, class_labels)

    # Create results dictionary
    results = {
        "name" : architectures[model_key]["name"],
        "train_results" : train_results,
        "test_results" : test_results
    }
    pprint(results)

    # Save results dictionary as a json file
    if info_path is not None:
        with open(info_path, "w") as out:
            out.write(json.dumps(results, indent=4))
