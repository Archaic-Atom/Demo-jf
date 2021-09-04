#!/bin/bash
CUDA_VISIBLE_DEVICES=6,7 nohup python -u Source/main.py \
                        --batchSize 2 \
                        --gpu 4 \
                        --trainListPath ./Datasets/kitti2012_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 4 \
                        --maxEpochs 100 \
                        --modelDir ./Checkpoint_kitti2012/ \
                        --auto_save_num 50 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --outputDir ./Result_kitti2012/ \
                        --modelName STTStereo \
                        --dataset kitti2012 > ./log/TrainRun_kitti2012.log 2>&1 &
