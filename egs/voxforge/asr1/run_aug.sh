#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=tdnn # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers

tdnn_offsets="-1,0,1 -1,0,1 -1,0,1 -3,0,3 -3,0,3 -3,0,3 -3,0,3"
tdnn_odims="625 625 625 625 625 625 625"
tdnn_prefinal_affine_dim=625
tdnn_final_affine_dim=300

# decoder related
dlayers=1
dunits=300

# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=1.0
use_lm=false

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.15
minlenratio=0.6
ctc_weight=0.0
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'
decode_nj=50

# aug data
aug_use=1
aug_ratio=0.5
aug_pretrain=2000
aug_path=aug_ca.jitter

# gan
gan_weight=0.0
gan_smooth=0.3
gan_odim=128
gan_wsize=20
gan_only=false

# data
voxforge=downloads # original data directory to be stored
lang=pt # de, en, es, fr, it, nl, pt, ru

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_${lang}
train_dev=dt_${lang}
recog_set="dt_${lang} et_${lang}"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    local/getdata.sh ${lang} ${voxforge}
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    selected=${voxforge}/${lang}/extracted
    # Initial normalization of the data
    local/voxforge_data_prep.sh ${selected} ${lang}
    local/voxforge_format_data.sh ${lang}
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 30 data/all_${lang} exp/make_fbank/train_${lang} ${fbankdir}

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or no more than 0 characters
    remove_longshortdata.sh data/all_${lang} data/all_trim_${lang}

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_trim_${lang} data/tr_${lang} data/dt_${lang} data/et_${lang}
    rm -r data/all_trim_${lang}

    # compute global CMVN
    compute-cmvn-stats scp:data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{13,14,15,16}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{13,14,15,16}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/tr_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/dt_${lang}/feats.scp data/tr_${lang}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/tr_${lang}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/tr_${lang}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    for rtask in tr_${lang} dt_${lang} ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkjson.py ${feat_recog_dir}/feats.scp data/${rtask} ${dict} > ${feat_recog_dir}/data.${lang}.json
    done
fi

dict_aug=data/lang_1char/aug_tr_${lang}_input_units.txt
if [ $stage -le 3 ]; then
    if [ ! -z "${aug_path}" ] && [ ${aug_use} -eq 1 ]; then
      echo "creating ${feat_tr_dir}/aug.json with augmenting data"
      aug2json.py -a ${aug_path} -d ${dict} -j ${feat_tr_dir}/aug.json > $dict_aug
    else
      echo "skipping data augmentation..."
    fi
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if [ "${aug_use}" -eq 1 ]; then
      expdir=${expdir}_aug_properties
      expdir=${expdir}_ratio${aug_ratio}_pre${aug_pretrain}_alt${aug_alternate}_arch${aug_arch}_layers${aug_layers}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    extra_opts=""
    if $gan_only; then
      extra_opts="--gan-only"
    fi
   
    echo "GAN ONLY: ${extra_opts}"
    
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.${lang}.json \
        --valid-json ${feat_dt_dir}/data.${lang}.json \
        --aug-use ${aug_use} \
        --aug-ratio ${aug_ratio} \
        --aug-pretrain ${aug_pretrain} \
        --aug-train ${feat_tr_dir}/aug.json \
        --aug-dict ${dict_aug} \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --tdnn-offsets "${tdnn_offsets}" \
        --tdnn-odims "${tdnn_odims}" \
        --tdnn-prefinal-affine-dim ${tdnn_prefinal_affine_dim} \
        --tdnn-final-affine-dim ${tdnn_final_affine_dim} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --gan-weight ${gan_weight} \
        --gan-smooth ${gan_smooth} \
        --gan-wsize ${gan_wsize} \
        --gan-odim ${gan_odim} \
        ${extra_opts}
fi


if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"

    extra_opts=""
    if $use_lm; then
      extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best --lm-weight ${lm_weight} ${extra_opts}"
    fi

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${decode_nj} ${feat_recog_dir}/data.${lang}.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${decode_nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${decode_nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.json \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --ctc-weight ${ctc_weight} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            ${extra_opts} &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

