#!/usr/bin/env python3

import os
import re
import sys
import json
import pickle
from math import ceil
from datetime import datetime

from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

import tensorflow_model_optimization.sparsity.keras as sparsity

import configs.pruning as p_config
from configs.architectures import architectures
from utils import cifar10
from utils.metrics import count_nonzero_params

def prune_layer(layer, pruning_schedule, exclude=None):
    if exclude is None or not re.match(exclude, layer.name):
        return sparsity.prune_low_magnitude(
            to_prune=layer, 
            pruning_schedule=pruning_schedule
        )
    return layer


if __name__ == "__main__":

    ### Get all paths of the unpruned and pruned model
    model_key = sys.argv[1]
    sparsity_key = sys.argv[2]
    unpruned_model_path = sys.argv[3]
    saved_model_dir = sys.argv[4]
    
    s_config = p_config.sparsity[sparsity_key]

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    pruned_model_base_dir = os.path.join(saved_model_dir, "{}_s{}_{}".format(model_key, s_config["final_sparsity"], date))
    pruned_model_path = os.path.join(pruned_model_base_dir, "{}_pruned.h5".format(model_key))
    history_path = os.path.join(pruned_model_base_dir, "train_history.pkl")
    info_path = os.path.join(pruned_model_base_dir, "info.json")
    tb_log_dir = os.path.join(pruned_model_base_dir, "logs", "training")


    ### Load and preprocess data
    preprocess_input = architectures[model_key]["preprocessing_fct"]
    (X_train, y_train), (X_test, y_test), _ = cifar10.load_and_preprocess(preprocess_input)


    ### Load model
    model = models.load_model(unpruned_model_path)


    ### Prune the model
    batch_size = 64
    epochs = s_config["start_pruning"] + s_config["pruning_steps"] * s_config["frequency"]
    steps_per_epoch = ceil(len(X_train) // batch_size + 1)


    # Define pruning schedule
    pruning_schedule = sparsity.PolynomialDecay(
        initial_sparsity=s_config["initial_sparsity"],
        final_sparsity=s_config["final_sparsity"],
        begin_step=s_config["start_pruning"]*steps_per_epoch,
        end_step=epochs*steps_per_epoch,
        frequency=s_config["frequency"]*steps_per_epoch
    )

    # Mark all layers matched a regex for pruning
    exclude_regex = p_config.exclude_regex[model_key]
    pruning_model = models.clone_model(
        model=model, 
        clone_function=lambda layer: prune_layer(layer, pruning_schedule, exclude_regex)
    )

    # Compile the model for pruning
    pruning_model.compile(
        optimizer=optimizers.RMSprop(
            learning_rate=optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-4,
                end_learning_rate=1e-5,
                decay_steps=epochs*steps_per_epoch,
                power=1
            ),
            momentum=0.9
        ),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.TopKCategoricalAccuracy(k=1, name="acc_top1"),
            metrics.TopKCategoricalAccuracy(k=3, name="acc_top3"),
        ]
    )

    # Train the model and apply pruning
    train_history = pruning_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            sparsity.UpdatePruningStep(),
            sparsity.PruningSummaries(log_dir=tb_log_dir)
        ]
    )

    # Remove prune_low_magnitude wrappers
    pruning_model = sparsity.strip_pruning(pruning_model)

    # Save model
    pruning_model.save(pruned_model_path)

    # Save train history
    with open(history_path, "wb") as out:
        pickle.dump(train_history.history, out)


    ### Evaluate
    # Compile the model for evaluation
    pruning_model.compile(
        optimizer=optimizers.RMSprop(
            learning_rate=optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-4,
                end_learning_rate=1e-5,
                decay_steps=epochs*steps_per_epoch,
                power=1
            ),
            momentum=0.9
        ),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.TopKCategoricalAccuracy(k=1, name="acc_top1"),
            metrics.TopKCategoricalAccuracy(k=3, name="acc_top3"),
        ]
    )

    # Save model informations in a dictionary
    info = dict()
    info["name"] = architectures[model_key]["name"]
    info["params"] = pruning_model.count_params()
    info["nonzero_params"] = count_nonzero_params(pruning_model)
    info["sparsity"] = s_config["final_sparsity"]
    info["actual_sparsity"] = (info["params"] - info["nonzero_params"]) / info["params"]

    # Evaluate on train and test data
    print("### Evaluate:", pruned_model_path)

    print("# On train data...")
    metrics_values = pruning_model.evaluate(X_train, y_train)
    info["train_results"] = dict(zip(pruning_model.metrics_names, metrics_values))

    print("# On test data...")
    metrics_values = pruning_model.evaluate(X_test, y_test)
    info["test_results"] = dict(zip(pruning_model.metrics_names, metrics_values))

    # Write info dictionary to a .json file
    with open(info_path, "w") as out:
        out.write(json.dumps(info, indent=4))
