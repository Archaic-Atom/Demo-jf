#!/bin/bash
rm -r ResultImg/
CUDA_VISIBLE_DEVICES=7 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 4 \
                        --trainListPath ./Datasets/middlebury_training_H_list.csv \
                        --imgWidth 1536 \
                        --imgHeight 1024 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 15 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --modelName STTStereo \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset middlebury
cp -r ResultImg/ trainingH/
zip -r trainingH.zip trainingH
mv trainingH.zip  ResultImg/
rm -r trainingH