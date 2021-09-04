#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 4 \
                        --trainListPath ./Datasets/kitti2012_testing_list.csv \
                        --imgWidth 1536 \
                        --imgHeight 512 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 194 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --modelName STTStereo \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint_kitti2012/ \
                        --dataset kitti2012
