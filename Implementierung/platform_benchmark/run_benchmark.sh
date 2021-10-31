#!/bin/bash

TOOL=./benchmark_model_aarch64

$TOOL --graph="$1" --num_threads=1 --warmup_runs=1 --num_runs=50