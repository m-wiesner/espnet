#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import codecs
import imageio 
import json
import argparse
import numpy as np
import glob
import os

# chainer related
import chainer
from chainer.datasets import TransformDataset

# torch related
import torch

# spnet related
from asr_utils import load_inputs_and_targets_augment
from asr_utils import get_model_conf
from asr_utils import torch_load
from asr_utils import make_augment_batchset
from asr_pytorch import AugmentConverter
from e2e_asr_th import Generator

# matplotlib related
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('model_conf', type=str)
    parser.add_argument('aug_json', type=str)
    parser.add_argument("--duration", type=float, action="store", default=1.0) 
    parser.add_argument("--epoch", type=str, action="store", default='')
    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.output)):
        os.makedirs(os.path.abspath(args.output))

    with codecs.open(args.aug_json, 'rb', encoding='utf-8') as f:
            augment_json = json.load(f)['aug']

    if args.epoch == '':
        models = glob.glob(args.model + '/results/gen.loss.best_*')
    else:
        epochs = [e for e in args.epoch.split()] 
        models = [args.model + '/results/gen.loss.best_' + e for e in epochs]

    idim, odim, train_args = get_model_conf(models[0], args.model_conf)
    train_args.gtype = 'tdnn'

    train_augment, meta = make_augment_batchset(augment_json, 1,
                                                    train_args.maxlen_in,
                                                    train_args.maxlen_out,
                                                    1,
                                                    train_args.subsample)
    converter_augment = AugmentConverter(meta['idict'], meta['odict'],
                                             meta['ifilename'], meta['ofilename'],
                                             expand_iline=3,
                                             subsamping_factor=1) 
    
    gen = Generator(train_args.aug_vocab_size, train_args.aug_idim, idim, train_args)
    train_augment_iter = chainer.iterators.SerialIterator(TransformDataset(train_augment, converter_augment.transform), batch_size=1, repeat=False, shuffle=False)
    ifile = codecs.open(meta['ifilename'], 'rb', encoding='utf-8')
    ofile = codecs.open(meta['ofilename'], 'rb', encoding='utf-8')
    b = train_augment_iter.__next__()
    x = converter_augment(b, 'cpu')

    for m in models:
        epoch = m.strip().split('_')[-1]
        torch_load(m, gen) 
        x_, _ = gen(x[0], x[1])
        np.save(args.output + '/output.' + epoch, x_.detach().numpy())
    
    # Make GIF
    files = glob.glob(args.output + "/output*.npy")
    files = sorted(files, key=lambda x : int(x.split('.')[-2]))
    images = []
    for f in files:
        fname = os.path.basename(f)
        out = np.load(f)
        plt.imshow(out[0].T, origin='lower', extent=(-3000, 3000, -1000, 1000))
        plt.savefig(args.output + "/" + fname + ".png")
        images.append(imageio.imread(args.output + "/" + fname + ".png"))
    imageio.mimsave(args.output + '/gen.gif', images, duration=args.duration)

if __name__ == "__main__":
    main()

