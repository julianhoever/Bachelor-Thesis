import tensorflow as tf


architectures = {
    "mobilenetv1" : {
        "name" : "MobileNetV1",
        "model" : tf.keras.applications.MobileNet,
        "model_args" : {
            "alpha" : 1,
            "depth_multiplier" : 1,
            "weights" : None
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet.preprocess_input
    },

    "mobilenetv2" : {
        "name" : "MobileNetV2",
        "model" : tf.keras.applications.MobileNetV2,
        "model_args" : {
            "alpha" : 1,
            "weights" : None
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet_v2.preprocess_input
    },

    "mobilenetv3_large" : {
        "name" : "MobileNetV3 Large",
        "model" : tf.keras.applications.MobileNetV3Large,
        "model_args" : {
            "alpha" : 1,
            "weights" : None
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet_v3.preprocess_input
    },

    "mobilenetv3_small" : {
        "name" : "MobileNetV3 Small",
        "model" : tf.keras.applications.MobileNetV3Small,
        "model_args" : {
            "alpha" : 1,
            "weights" : None
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet_v3.preprocess_input
    },

    "efficientnet-b0" : {
        "name" : "EfficientNet-B0",
        "model" : tf.keras.applications.EfficientNetB0,
        "model_args" : {
            "weights" : None
        },
        "preprocessing_fct" : tf.keras.applications.efficientnet.preprocess_input
    },

    "mobilenetv3_large_minimalistic" : {
        "name" : "MobileNetV3 Large (minimalistic)",
        "model" : tf.keras.applications.MobileNetV3Large,
        "model_args" : {
            "alpha" : 1,
            "weights" : None,
            "minimalistic" : True
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet_v3.preprocess_input
    },

    "mobilenetv3_small_minimalistic" : {
        "name" : "MobileNetV3 Small (minimalistic)",
        "model" : tf.keras.applications.MobileNetV3Small,
        "model_args" : {
            "alpha" : 1,
            "weights" : None,
            "minimalistic" : True
        },
        "preprocessing_fct" : tf.keras.applications.mobilenet_v3.preprocess_input
    }
}