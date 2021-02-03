# A2W Segmental Model (SLT'2021)
This repo contains the code for A2W segmental model [Whole-Word Segmental Speech Recognition with Acoustic Word Embeddings](https://arxiv.org/pdf/2007.00183.pdf)

## Requirements
* Kaldi
* g++ 5.5.0
* CUDA 10.0
* Pytorch 1.4.0
* kaldi-io

## Usage:
1. Use Kaldi for data preparation: setting up data dir, extracting fbank feature, computing and applying CMVN. 
3. Extract word list and map words into ids. This can be done via `src/preproc/prep_seg_word.py`.
3. Edit paths for feature and text data in the configuration file (in conf/). An example of data and configuration file can be found in `data/` and `src/conf/`.

4.  Training and evaluation
```sh
cd src
./run.sh
```
! Note before running `./run.sh`, make sure following paths are correctly set including `KALDI_ROOT` in `path.sh`, `CC,CXX,PATH,LD_LIBRARY_PATH` in `run.sh` 

## ToDo:
- [X] *Code release for segmental model*
- [] Code release for AWE/AGWE pre-training


## Reference

Please cite the paper below if you use this code in your research:

@inproceedings{shi2021segmental,
   author = {Bowen Shi and Shane Settle and Karen Livescu},
   title = {Whole-Word Segmental Speech Recognition with Acoustic Word Embeddings},
   booktitle = {SLT},
   year = {2021}
}
