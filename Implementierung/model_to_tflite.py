#!/usr/bin/env python3

import sys

import tensorflow as tf
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    # Save commandline arguments
    model_path = sys.argv[1]
    tflite_path = sys.argv[2]

    # Load specified model
    model = load_model(model_path)
    
    # Convert model to tflite representation
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save tflite model to file
    with open(tflite_path, "wb") as out:
        out.write(tflite_model)
