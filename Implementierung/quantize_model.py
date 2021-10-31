#!/usr/bin/env python3

import sys
import tensorflow as tf

from tensorflow.keras.models import load_model

from utils import cifar10
from configs.architectures import architectures


if __name__ == "__main__":

    ### Get all commandline arguments
    model_key = sys.argv[1]
    model_path = sys.argv[2]
    tflite_path = sys.argv[3]
    int_only = len(sys.argv) > 4 and sys.argv[4] == "int_only"


    ### Load and preprocess data
    preprocess_input = architectures[model_key]["preprocessing_fct"]
    (X_train, _), (X_test, _), _ = cifar10.load_and_preprocess(preprocess_input, dtype_X="float32")


    ### Load saved model
    model = load_model(model_path)


    ### Quantize model
    # Representative subset of the dataset to learn typical quantization ranges
    def representative_dataset():
        for img in X_train[:100]:
            yield [ img.reshape((1, *img.shape)) ]

    # Quantize loaded model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [ tf.lite.Optimize.DEFAULT ]
    converter.representative_dataset = representative_dataset
    if int_only:
        converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS_INT8 ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    # Save tflite model to file
    with open(tflite_path, "wb") as out:
        out.write(tflite_model)
