#!/bin/bash

PRETRAINED=saved_models/pretrained

python3 train_model.py mobilenetv1 $PRETRAINED
python3 train_model.py mobilenetv2 $PRETRAINED
python3 train_model.py mobilenetv3_large $PRETRAINED
python3 train_model.py mobilenetv3_small $PRETRAINED
python3 train_model.py efficientnet-b0 $PRETRAINED

# Minimalistic MobileNetV3
python3 train_mnv3_minimalistic.py mobilenetv3_large_minimalistic $PRETRAINED
python3 train_mnv3_minimalistic.py mobilenetv3_small_minimalistic $PRETRAINED
