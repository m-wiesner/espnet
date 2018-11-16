#!/usr/bin/env python

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import platform
import random
import subprocess
import sys
import codecs

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str)
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--dict', required=True,
                        help='Dictionary')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--debugdir', type=str,
                        help='Output directory for debugging')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--train-json', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    # network archtecture
    # encoder
    parser.add_argument('--etype', default='blstmp', type=str,
                        choices=['tdnn', 'blstm', 'blstmp', 'vggblstmp', 'vggblstm'],
                        help='Type of encoder network architecture')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=300, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=320, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default=1, type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')  
    # Encoder TDNN 
    parser.add_argument('--tdnn-odims', default='', type=str,
                        action='store', required=False, help='specify the output dimension of each tdnn layer')
    parser.add_argument('--tdnn-offsets', default='', type=str,
                        action='store', required=False, help='specify the offsets for each layer in the tdnn')
    parser.add_argument('--tdnn-prefinal-affine-dim', default=625, type=int,
                        action='store', required=False,
                        help='tdnn prefinal affine dimension')
    parser.add_argument('--tdnn-final-affine-dim', default=300, type=int,
                        action='store', required=False,
                        help='tdnn final affine dimension')

    # loss
    parser.add_argument('--ctc_type', default='warpctc', type=str,
                        choices=['chainer', 'warpctc'],
                        help='Type of CTC implementation to calculate loss.')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                 'coverage_location', 'location2d', 'location_recurrent',
                                 'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                 'multi_head_multi_res_loc'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=320, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--awin', default=5, type=int,
                        help='Window size for location2d attention')
    parser.add_argument('--aheads', default=4, type=int,
                        help='Number of heads for multi head attention')
    parser.add_argument('--aconv-chans', default=-1, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--aconv-filts', default=100, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    # decoder
    parser.add_argument('--dtype', default='lstm', type=str,
                        choices=['lstm'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=320, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--mtlalpha', default=0.5, type=float,
                        help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
    parser.add_argument('--mtlalpha-decay', default=1.0, type=float,
                        help='Multitask learning coefficient decay rate, alpha *= decay')
    parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                        help='Apply label smoothing with a specified distribution type')
    parser.add_argument('--lsm-weight', default=0.0, type=float,
                        help='Label smoothing weight')
    parser.add_argument('--sampling-probability', default=0.0, type=float,
                        help='Ratio of predicted labels fed back to decoder')

    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate')
    # minibatch related
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam', 'datadelta'],
                        help='Optimizer')
    parser.add_argument('--datadelta-check-batch-types', default='ctc', type=str,
                        help='which batches to check (only used in datadelta optimizer)')
    parser.add_argument('--datadelta-undo-type', default='undo_none', type=str,
                        choices=set(['undo_step', 'undo_learning_rate', 'undo_square_avg', 'undo_none']),
                        help='type of undo mechanism to use.')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=3, type=int,
                        help='Number of samples of attention to be saved')

    # Aug related
    parser.add_argument('--aug-use', type=int, choices=set([0, 1]), default=0,
                        help='enables or disables augment based training')
    parser.add_argument('--aug-train', type=str, required=False, default='', nargs='?',
                        help='Filename of aug data (json)')
    parser.add_argument('--aug-dict', type=str, required=False, default='', nargs='?',
                        help='Filename of aug input dictionary')
    parser.add_argument('--aug-ratio', default=0.5, type=float,
                        help='aug to real data ratio (smaller number means less aug batches)')
    parser.add_argument('--aug-pretrain', default=0, type=int,
                        help='number of aug batches before audio batches')
    parser.add_argument('--aug-idim', default=83, type=int,
                        help='augment symbol embedding dim')
    parser.add_argument('--auglayers', default=1, type=int)
    parser.add_argument('--augunits', default=300, type=int)
    parser.add_argument('--augprojs', default=300, type=int)
     
    
    # GAN related
    parser.add_argument('--gan-weight', default=0.0, type=float, help='Amount of GAN loss')
    parser.add_argument('--gan-smooth', default=0.3, type=float, help='Amount of GAN loss')
    parser.add_argument('--gan-wsize', default=20, type=int, help='Window size for GAN loss')
    parser.add_argument('--gan-odim', default=128, type=int, help='Dimension of discriminator outputs (prefinal layer)') 
    parser.add_argument('--gan-only', action='store_true', help='Train the GAN only')
    parser.add_argument('--gtype', default='blstmp', type=str,
                        choices=['tdnn', 'blstm', 'blstmp', 'vggblstmp', 'vggblstm'],
                        help='Type of encoder network architecture')
 
    
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == '2':
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    if args.aug_dict != '' and args.aug_use == 1:
        with codecs.open(args.aug_dict, 'r', encoding='utf-8') as f:
            aug_dictionary = f.readlines()
        args.aug_vocab_size = len(aug_dictionary)
    else:
        args.aug_vocab_size = None

    # train
    from asr_pytorch_diagnostic import train
    train(args)


if __name__ == '__main__':
    main()


