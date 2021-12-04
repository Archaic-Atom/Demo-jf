#!/bin/bash
# parameters
tensorboard_port=6235
dist_port=8801
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u Source/main.py \
                        --batchSize 256\
                        --gpu 8 \
                        --trainListPath ./Datasets/mnist_dataset.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 24 \
                        --maxEpochs 200 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist True \
                        --modelName ConvNet \
                        --port ${dist_port} \
                        --dataset mnist > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"
echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ./log --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

echo "Begin to watch TrainRun.log file!"
tail -f TrainRun.log