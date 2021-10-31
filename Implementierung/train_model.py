#!/usr/bin/env python3

import os
import sys
import json
import pickle
from math import ceil
from datetime import datetime

from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks

from configs.architectures import architectures
from utils import cifar10
from utils.metrics import count_nonzero_params


if __name__ == "__main__":

    ### Load config
    # Get selected architecture key
    model_key = sys.argv[1]
    saved_model_dir = sys.argv[2]

    # Check if selected architecture is configured in config.used_architectures
    available_keys = list(architectures.keys())
    if model_key not in available_keys:
        print("Please select an configured architecture for training:", available_keys)
        exit(-1)

    # Extract neccessary informations from config file
    model_name = architectures[model_key]["name"]
    model_class = architectures[model_key]["model"]
    model_args = architectures[model_key]["model_args"]
    preprocess_input = architectures[model_key]["preprocessing_fct"]

    # Create paths to save data for this training run
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_base_dir = os.path.join(saved_model_dir, "{}_{}".format(model_key, date))
    model_path = os.path.join(model_base_dir, "{}_pretrained.h5".format(model_key))
    history_path = os.path.join(model_base_dir, "train_history.pkl")
    info_path = os.path.join(model_base_dir, "info.json")
    tb_log_dir = os.path.join(model_base_dir, "logs", "training")


    ### Load and preprocess data
    preprocess_input = architectures[model_key]["preprocessing_fct"]
    (X_train, y_train), (X_test, y_test), _ = cifar10.load_and_preprocess(preprocess_input)

    # Get shape of a single image and number of classes
    img_height, img_width, img_channel = X_train.shape[1:]
    n_classes = y_train.shape[1]


    ### Create and train model
    epochs = 150
    batch_size = 64
    n_steps = ceil(len(X_train) // batch_size + 1) * epochs

    # Create model
    model = model_class(
        input_shape=(img_height, img_width, img_channel),
        classes=n_classes,
        **model_args
    )

    # Compile model
    model.compile(
        optimizer=optimizers.RMSprop(
            learning_rate=optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-3,
                end_learning_rate=0,
                decay_steps=n_steps,
                power=1
            ),
            momentum=0.9
        ),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.TopKCategoricalAccuracy(k=1, name="acc_top1"),
            metrics.TopKCategoricalAccuracy(k=3, name="acc_top3")
        ]
    )

    # Train model
    train_history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=50,
                restore_best_weights=True
            ),
            callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
    )

    # Save model
    model.save(model_path)

    # Save train history
    with open(history_path, "wb") as out:
        pickle.dump(train_history.history, out)


    ### Evaluate model
    # Load model
    model = models.load_model(model_path)

    # Save model informations in a dictionary
    info = dict()
    info["name"] = model_name
    info["params"] = model.count_params()
    info["nonzero_params"] = count_nonzero_params(model)
    info["sparsity"] = 0
    info["actual_sparsity"] = (info["params"] - info["nonzero_params"]) / info["params"]

    # Evaluate on train and test data
    print("### Evaluate:", model_path)

    print("# On train data...")
    metrics_values = model.evaluate(X_train, y_train)
    info["train_results"] = dict(zip(model.metrics_names, metrics_values))

    print("# On test data...")
    metrics_values = model.evaluate(X_test, y_test)
    info["test_results"] = dict(zip(model.metrics_names, metrics_values))

    # Write info dictionary to a .json file
    with open(info_path, "w") as out:
        out.write(json.dumps(info, indent=4))
