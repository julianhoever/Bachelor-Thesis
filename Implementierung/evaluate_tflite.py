#!/usr/bin/env python3

import sys
import json
import numpy as np
from pprint import pprint

from tensorflow.keras.metrics import categorical_crossentropy, top_k_categorical_accuracy

from configs.architectures import architectures
from utils import cifar10, tflite


def evaluate_tflite(model_path, metrics_dict, samples, labels):
    # Get predictions
    predictions = tflite.predict(model_path, samples)
    
    # Evaluate metrics on the predicted labels
    scores = dict()
    for metric_name, metric in metrics_dict.items():
        metric_scores = metric(labels, predictions)
        metric_mean = np.mean(metric_scores)
        scores[metric_name] = float(metric_mean)

    return scores


if __name__ == "__main__":

    ### Extract command line arguments
    model_key = sys.argv[1]
    tflite_path = sys.argv[2]
    info_path = sys.argv[3] if len(sys.argv) > 3 else None


    ### Load and preprocess data
    preprocess_input = architectures[model_key]["preprocessing_fct"]
    (X_train, y_train), (X_test, y_test), _ = cifar10.load_and_preprocess(preprocess_input)


    ### Evaluate metrics
    # Define metrics
    metrics_dict = {
        "loss" : categorical_crossentropy,
        "acc_top1" : lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=1),
        "acc_top3" : lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=3)
    }

    # Evaluate tflite file on train and test set
    evaluate = lambda X, y: evaluate_tflite(tflite_path, metrics_dict, X, y)
    train_results = evaluate(X_train, y_train)
    test_results = evaluate(X_test, y_test)

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