#!/bin/bash


URL=http://neuralvfx.com/datasets/light_swap/wat_mai_amataros.rar
ZIP_FILE=./data/wat_mai_amataros.rar
TARGET_DIR=./data/wat_mai_amataros/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unrar x $ZIP_FILE  $TARGET_DIR
