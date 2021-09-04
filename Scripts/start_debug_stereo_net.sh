#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -u Source/main.py \
                        --mode train \
                        --batchSize 64 \
                        --gpu 2 \
                        --trainListPath ./Datasets/scene_flow_debug_training_list.csv \
                        --valListPath ./Datasets/scene_flow_debug_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 0 \
                        --maxEpochs 20 \
                        --imgNum 35454 \
                        --valImgNum 2003 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist True \
                        --modelName DPFNN \
                        --outputDir ./DebugResult/ \
                        --modelDir ./DebugCheckpoint/ \
                        --dataset sceneflow
