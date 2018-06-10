#!/bin/bash

URL=https:http://neuralvfx.com/datasets/light_swap/wat_mai_amataros.rar
ZIP_FILE=./data/wat_mai_amataros.rar
TARGET_DIR=./data/wat_mai_amataros/
wget -N $URL -O $ZIP_FILE
mkdir wat_mai_amataros
unrar $ZIP_FILE -x $TARGET_DIR=
rm $ZIP_FILE