#!/bin/bash
CUDA_VISIBLE_DEVICES=7  python Source/main.py \
                        --mode test \
                        --outputDir ./TestResult/ \
                        --batchSize 64 \
                        --trainListPath ./Datasets/msd_val_list_2017.csv \
                        --imgNum 4500