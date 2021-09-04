#!/bin/bash
rm -r ResultImg/
CUDA_VISIBLE_DEVICES=7 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 4 \
                        --trainListPath ./Datasets/eth3d_testing_list.csv \
                        --imgWidth 1024 \
                        --imgHeight 768 \
                        --dataloaderNum 16 \
                        --maxEpochs 45 \
                        --imgNum 27 \
                        --sampleNum 1 \
                        --lr 0.0001 \
                        --modelName STTStereo \
                        --outputDir ./DebugResult/ \
                        --modelDir ./Checkpoint/ \
                        --dataset eth3d
cp -r ResultImg/ low_res_two_view/
zip -r low_res_two_view.zip low_res_two_view
mv low_res_two_view.zip  ResultImg/
rm -r low_res_two_view/