#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -u Source/main.py \
						--batchSize 2 \
						--gpu 4 \
						--trainListPath ./Datasets/dfc_training_list.csv \
						--imgWidth 480 \
						--imgHeight 384 \
						--dataloaderNum 16 \
						--maxEpochs 500 \
						--imgNum 4500 \
						--dataset US3D
