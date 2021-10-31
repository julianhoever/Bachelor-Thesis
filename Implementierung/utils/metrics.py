from typing import List, Dict
import numpy as np
import tensorflow as tf


def count_nonzero_params(model: tf.keras.models.Model) -> int:
    return int(sum(tf.math.count_nonzero(W) for W in model.get_weights()))


def top_k_classwise_accuracy(
        y_true: List[int], 
        y_pred: List[np.ndarray], 
        k: int=1, 
        class_labels: 
        List[str]=None
    ) -> Dict[str, float]:
    # Array to store predictions for each class
    correct_predicted = [ list() for _ in range(len(y_true[0])) ]
    
    # For each class calculate if correct predicted
    for true, pred in zip(y_true, y_pred):
        top_k_pred_classes = np.argsort(pred)[-k:]
        true_class = np.argmax(true)
        correct_predicted[true_class].append(1 if true_class in top_k_pred_classes else 0)

    # Calculate accuracy for each class
    classwise_accuracy = np.mean(correct_predicted, axis=1)

    # If given add the class label to each accuracy
    if class_labels is not None:
        classwise_accuracy = dict(zip(class_labels, classwise_accuracy))

    # Return the accuracy for each class
    return classwise_accuracy