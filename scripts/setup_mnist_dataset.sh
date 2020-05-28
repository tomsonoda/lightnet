#!/bin/sh

TRAIN_DIR="dataset/mnist/train";
TEST_DIR="dataset/mnist/test";

if [ ! -d $TRAIN_DIR ]; then
    echo "Create $TRAIN_DIR";
    mkdir -p $TRAIN_DIR;
fi

if [ ! -d $TEST_DIR ]; then
    echo "Create $TEST_DIR";
    mkdir -p $TEST_DIR;
fi

wget -P $TRAIN_DIR http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P $TRAIN_DIR http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P $TEST_DIR http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P $TEST_DIR http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip $TRAIN_DIR/*.gz;
gunzip $TEST_DIR/*.gz;
