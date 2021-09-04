#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 nohup python -u Source/main.py \
                        --batchSize 2 \
                        --gpu 4 \
                        --trainListPath ./Datasets/dfc_training_list.csv \
                        --dataloaderNum 16 \
                        --maxEpochs 500 \
                        --imgNum 4500 > TrainRun.log 2>&1 &