sparsity = {
    "30%" : {
        "initial_sparsity" : 0,
        "final_sparsity" : 0.3,
        "start_pruning" : 0,
        "frequency" : 1,
        "pruning_steps" : 2
    },
    "60%" : {
        "initial_sparsity" : 0,
        "final_sparsity" : 0.6,
        "start_pruning" : 0,
        "frequency" : 2,
        "pruning_steps" : 3
    },
    "90%" : {
        "initial_sparsity" : 0,
        "final_sparsity" : 0.9,
        "start_pruning" : 0,
        "frequency" : 6,
        "pruning_steps" : 4
    }
}

exclude_regex = {
    "mobilenetv1" : None,
    "mobilenetv2" : None,
    "mobilenetv3_large" : r"^(rescaling|tf\.)",
    "mobilenetv3_small" : r"^(rescaling|tf\.)",
    "efficientnet-b0" : r"^(rescaling|normalization)",

    "mobilenetv3_large_minimalistic" : r"^(rescaling|tf\.)",
    "mobilenetv3_small_minimalistic" : r"^(rescaling|tf\.)"
}