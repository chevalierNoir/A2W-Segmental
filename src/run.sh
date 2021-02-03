#! /bin/bash
export CUDA_HOME=/usr/local/cuda/cuda-10.0/
export CXX=/usr/local/gcc-5.5.0/bin/g++
export CC=/usr/local/gcc-5.5.0/bin/gcc
export PATH=$PATH:$CUDA_HOME/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/cuda-10.0/lib64/:/usr/local/gcc-5.5.0/lib64/
export PYTHONPATH=$PYTHONPATH:./segloss
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/release-v2/
. ./path.sh

echo "Train"
python -B train_seg.py --config ./conf/train.yaml --output ../data/exp || exit 1;

echo "Eval"
python eval_seg.py --config_train ../data/exp/train_conf.yaml --config_eval ./conf/eval.yaml || exit 1;
