#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -u Source/main.py \
                        --mode test \
                        --batchSize 4 \
                        --gpu 4 \
                        --trainListPath ./Datasets/kitti2015_training_list.csv \
                        --imgWidth 1536 \
                        --imgHeight 512 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 200 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --modelName STTStereo \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset kitti2015
