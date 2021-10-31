from typing import List

import numpy as np
import tensorflow as tf


def predict(model_path: str, samples: np.ndarray) -> List[np.ndarray]:
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output informations
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_tensor = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0]['quantization']
    output_tensor = output_details[0]['index']
    output_quant = output_details[0]["quantization"]

    # Get prediction for every sample in samples
    predictions = []
    for sample in samples:
        # Quantize inputs if model is integer only quantized
        if input_dtype == np.uint8:
            scale, zero_point = input_quant
            sample = sample / scale + zero_point

        sample = np.expand_dims(sample, axis=0).astype(input_dtype)
        interpreter.set_tensor(input_tensor, sample)
        interpreter.invoke()
        inference = interpreter.get_tensor(output_tensor)

        # Convert inference back to float32 for evaluation
        # (not optimal for integer only hardware)
        if inference.dtype == np.uint8:
            scale, zero_point = output_quant
            inference = scale * (inference.astype("float32") - zero_point)

        predictions.append(inference.reshape((-1,)))

    return predictions
