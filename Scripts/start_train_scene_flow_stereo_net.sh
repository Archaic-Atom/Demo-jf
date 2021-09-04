#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u Source/main.py \
                        --batchSize 4\
                        --gpu 4 \
                        --trainListPath ./Datasets/scene_flow_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 16 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.005 \
                        --dist True \
                        --modelName STTStereo_v2 \
                        --dataset sceneflow > TrainRun.log 2>&1 &
