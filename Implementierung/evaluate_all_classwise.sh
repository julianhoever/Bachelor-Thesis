#!/bin/bash

# Base directories for tflite models
INIT_DIR=tflite_files/initial
PRU_DIR=tflite_files/pruned
PRU_QUANT_DIR=tflite_files/pruned_quantized
QUANT_DIR=tflite_files/quantized

python3 evaluate_tflite_classwise.py mobilenetv1 $INIT_DIR/mobilenetv1.tflite $INIT_DIR/mobilenetv1_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $QUANT_DIR/mobilenetv1.tflite $QUANT_DIR/mobilenetv1_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_DIR/mobilenetv1_s30.tflite $PRU_DIR/mobilenetv1_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_DIR/mobilenetv1_s60.tflite $PRU_DIR/mobilenetv1_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_DIR/mobilenetv1_s90.tflite $PRU_DIR/mobilenetv1_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_QUANT_DIR/mobilenetv1_s30.tflite $PRU_QUANT_DIR/mobilenetv1_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_QUANT_DIR/mobilenetv1_s60.tflite $PRU_QUANT_DIR/mobilenetv1_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv1 $PRU_QUANT_DIR/mobilenetv1_s90.tflite $PRU_QUANT_DIR/mobilenetv1_s90_classwise.json

python3 evaluate_tflite_classwise.py mobilenetv2 $INIT_DIR/mobilenetv2.tflite $INIT_DIR/mobilenetv2_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $QUANT_DIR/mobilenetv2.tflite $QUANT_DIR/mobilenetv2_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_DIR/mobilenetv2_s30.tflite $PRU_DIR/mobilenetv2_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_DIR/mobilenetv2_s60.tflite $PRU_DIR/mobilenetv2_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_DIR/mobilenetv2_s90.tflite $PRU_DIR/mobilenetv2_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_QUANT_DIR/mobilenetv2_s30.tflite $PRU_QUANT_DIR/mobilenetv2_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_QUANT_DIR/mobilenetv2_s60.tflite $PRU_QUANT_DIR/mobilenetv2_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv2 $PRU_QUANT_DIR/mobilenetv2_s90.tflite $PRU_QUANT_DIR/mobilenetv2_s90_classwise.json

python3 evaluate_tflite_classwise.py mobilenetv3_large $INIT_DIR/mobilenetv3_large.tflite $INIT_DIR/mobilenetv3_large_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $QUANT_DIR/mobilenetv3_large.tflite $QUANT_DIR/mobilenetv3_large_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_DIR/mobilenetv3_large_s30.tflite $PRU_DIR/mobilenetv3_large_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_DIR/mobilenetv3_large_s60.tflite $PRU_DIR/mobilenetv3_large_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_DIR/mobilenetv3_large_s90.tflite $PRU_DIR/mobilenetv3_large_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_QUANT_DIR/mobilenetv3_large_s30.tflite $PRU_QUANT_DIR/mobilenetv3_large_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_QUANT_DIR/mobilenetv3_large_s60.tflite $PRU_QUANT_DIR/mobilenetv3_large_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large $PRU_QUANT_DIR/mobilenetv3_large_s90.tflite $PRU_QUANT_DIR/mobilenetv3_large_s90_classwise.json

python3 evaluate_tflite_classwise.py mobilenetv3_small $INIT_DIR/mobilenetv3_small.tflite $INIT_DIR/mobilenetv3_small_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $QUANT_DIR/mobilenetv3_small.tflite $QUANT_DIR/mobilenetv3_small_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_DIR/mobilenetv3_small_s30.tflite $PRU_DIR/mobilenetv3_small_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_DIR/mobilenetv3_small_s60.tflite $PRU_DIR/mobilenetv3_small_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_DIR/mobilenetv3_small_s90.tflite $PRU_DIR/mobilenetv3_small_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_QUANT_DIR/mobilenetv3_small_s30.tflite $PRU_QUANT_DIR/mobilenetv3_small_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_QUANT_DIR/mobilenetv3_small_s60.tflite $PRU_QUANT_DIR/mobilenetv3_small_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small $PRU_QUANT_DIR/mobilenetv3_small_s90.tflite $PRU_QUANT_DIR/mobilenetv3_small_s90_classwise.json

python3 evaluate_tflite_classwise.py efficientnet-b0 $INIT_DIR/efficientnet-b0.tflite $INIT_DIR/efficientnet-b0_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $QUANT_DIR/efficientnet-b0.tflite $QUANT_DIR/efficientnet-b0_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_DIR/efficientnet-b0_s30.tflite $PRU_DIR/efficientnet-b0_s30_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_DIR/efficientnet-b0_s60.tflite $PRU_DIR/efficientnet-b0_s60_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_DIR/efficientnet-b0_s90.tflite $PRU_DIR/efficientnet-b0_s90_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_QUANT_DIR/efficientnet-b0_s30.tflite $PRU_QUANT_DIR/efficientnet-b0_s30_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_QUANT_DIR/efficientnet-b0_s60.tflite $PRU_QUANT_DIR/efficientnet-b0_s60_classwise.json
python3 evaluate_tflite_classwise.py efficientnet-b0 $PRU_QUANT_DIR/efficientnet-b0_s90.tflite $PRU_QUANT_DIR/efficientnet-b0_s90_classwise.json

# Minimalistic MobileNetV3
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $INIT_DIR/mobilenetv3_large_minimalistic.tflite $INIT_DIR/mobilenetv3_large_minimalistic_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $QUANT_DIR/mobilenetv3_large_minimalistic.tflite $QUANT_DIR/mobilenetv3_large_minimalistic_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_DIR/mobilenetv3_large_minimalistic_s30.tflite $PRU_DIR/mobilenetv3_large_minimalistic_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_DIR/mobilenetv3_large_minimalistic_s60.tflite $PRU_DIR/mobilenetv3_large_minimalistic_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_DIR/mobilenetv3_large_minimalistic_s90.tflite $PRU_DIR/mobilenetv3_large_minimalistic_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s30.tflite $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s60.tflite $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_large_minimalistic $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s90.tflite $PRU_QUANT_DIR/mobilenetv3_large_minimalistic_s90_classwise.json

python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $INIT_DIR/mobilenetv3_small_minimalistic.tflite $INIT_DIR/mobilenetv3_small_minimalistic_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $QUANT_DIR/mobilenetv3_small_minimalistic.tflite $QUANT_DIR/mobilenetv3_small_minimalistic_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_DIR/mobilenetv3_small_minimalistic_s30.tflite $PRU_DIR/mobilenetv3_small_minimalistic_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_DIR/mobilenetv3_small_minimalistic_s60.tflite $PRU_DIR/mobilenetv3_small_minimalistic_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_DIR/mobilenetv3_small_minimalistic_s90.tflite $PRU_DIR/mobilenetv3_small_minimalistic_s90_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s30.tflite $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s30_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s60.tflite $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s60_classwise.json
python3 evaluate_tflite_classwise.py mobilenetv3_small_minimalistic $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s90.tflite $PRU_QUANT_DIR/mobilenetv3_small_minimalistic_s90_classwise.json
