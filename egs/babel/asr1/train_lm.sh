#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
verbose=0      # verbose option

# rnnlm related
lm_vocabsize=50000
lm_layers=1
lm_units=650
lm_opt=sgd        # or adam
lm_batchsize=256  # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_maxlen=40     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
lmdatadir=data/local/wordlm_train
lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: ./train_lm.sh <train> <dev> <test>"
  exit 1;
fi

train=$1
dev=$2
eval=$3

mkdir -p ${lmexpdir}
mkdir -p ${lmdatadir}

cat ${train} | cut -f 2- -d" " > ${lmdatadir}/train.txt
cat ${dev} | cut -f 2- -d" " > ${lmdatadir}/valid.txt
cat ${eval} | cut -f 2- -d" " > ${lmdatadir}/test.txt
text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt

# use only 1 gpu
if [ ${ngpu} -gt 1 ]; then
    echo "LM training does not support multi-gpu. signle gpu will be used."
fi
${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
    lm_train.py \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --verbose 1 \
    --outdir ${lmexpdir} \
    --train-label ${lmdatadir}/train.txt \
    --valid-label ${lmdatadir}/valid.txt \
    --test-label ${lmdatadir}/test.txt \
    --resume ${lm_resume} \
    --layer ${lm_layers} \
    --unit ${lm_units} \
    --opt ${lm_opt} \
    --batchsize ${lm_batchsize} \
    --epoch ${lm_epochs} \
    --maxlen ${lm_maxlen} \
    --dict ${lmdict}


