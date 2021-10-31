#!/bin/bash

PRUNED=saved_models/pruned

MOBILENETV1=saved_models/pretrained/mobilenetv1_20210203-080705/mobilenetv1_pretrained.h5
MOBILENETV2=saved_models/pretrained/mobilenetv2_20210203-085938/mobilenetv2_pretrained.h5
MOBILENETV3_LARGE=saved_models/pretrained/mobilenetv3_large_20210203-102728/mobilenetv3_large_pretrained.h5
MOBILENETV3_SMALL=saved_models/pretrained/mobilenetv3_small_20210203-121017/mobilenetv3_small_pretrained.h5
EFFICIENTNETB0=saved_models/pretrained/efficientnet-b0_20210203-132025/efficientnet-b0_pretrained.h5

MOBILENETV3_LARGE_MINI=saved_models/pretrained/mobilenetv3_large_minimalistic_20210224-114751/mobilenetv3_large_minimalistic_pretrained.h5
MOBILENETV3_SMALL_MINI=saved_models/pretrained/mobilenetv3_small_minimalistic_20210224-124800/mobilenetv3_small_minimalistic_pretrained.h5

python3 prune_model.py mobilenetv1 30% $MOBILENETV1 $PRUNED
python3 prune_model.py mobilenetv1 60% $MOBILENETV1 $PRUNED
python3 prune_model.py mobilenetv1 90% $MOBILENETV1 $PRUNED

python3 prune_model.py mobilenetv2 30% $MOBILENETV2 $PRUNED
python3 prune_model.py mobilenetv2 60% $MOBILENETV2 $PRUNED
python3 prune_model.py mobilenetv2 90% $MOBILENETV2 $PRUNED

python3 prune_model.py mobilenetv3_large 30% $MOBILENETV3_LARGE $PRUNED
python3 prune_model.py mobilenetv3_large 60% $MOBILENETV3_LARGE $PRUNED
python3 prune_model.py mobilenetv3_large 90% $MOBILENETV3_LARGE $PRUNED

python3 prune_model.py mobilenetv3_small 30% $MOBILENETV3_SMALL $PRUNED
python3 prune_model.py mobilenetv3_small 60% $MOBILENETV3_SMALL $PRUNED
python3 prune_model.py mobilenetv3_small 90% $MOBILENETV3_SMALL $PRUNED

python3 prune_model.py efficientnet-b0 30% $EFFICIENTNETB0 $PRUNED
python3 prune_model.py efficientnet-b0 60% $EFFICIENTNETB0 $PRUNED
python3 prune_model.py efficientnet-b0 90% $EFFICIENTNETB0 $PRUNED

# Minimalistic MobileNetV3
python3 prune_mnv3_minimalistic.py mobilenetv3_large_minimalistic 30% $MOBILENETV3_LARGE_MINI $PRUNED
python3 prune_mnv3_minimalistic.py mobilenetv3_large_minimalistic 60% $MOBILENETV3_LARGE_MINI $PRUNED
python3 prune_mnv3_minimalistic.py mobilenetv3_large_minimalistic 90% $MOBILENETV3_LARGE_MINI $PRUNED

python3 prune_mnv3_minimalistic.py mobilenetv3_small_minimalistic 30% $MOBILENETV3_SMALL_MINI $PRUNED
python3 prune_mnv3_minimalistic.py mobilenetv3_small_minimalistic 60% $MOBILENETV3_SMALL_MINI $PRUNED
python3 prune_mnv3_minimalistic.py mobilenetv3_small_minimalistic 90% $MOBILENETV3_SMALL_MINI $PRUNED