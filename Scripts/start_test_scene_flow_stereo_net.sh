#!/bin/bash
CUDA_VISIBLE_DEVICES=6,7 python -u Source/main.py \
                        --mode test \
                        --batchSize 2 \
                        --gpu 5 \
                        --trainListPath ./Datasets/scene_flow_testing_list.csv \
                        --imgWidth 1024 \
                        --imgHeight 640 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 4370 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --modelName STTStereo \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset sceneflow
