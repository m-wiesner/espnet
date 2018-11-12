#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import random
import codecs

# chainer related
import chainer

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import add_results_to_json
from asr_utils import CompareValueTrigger
from asr_utils import get_model_conf
from asr_utils import load_inputs_and_targets
from asr_utils import load_inputs_and_targets_augment
from asr_utils import make_batchset
from asr_utils import make_augment_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from asr_utils import torch_load
from asr_utils import torch_resume
from asr_utils import torch_save
from asr_utils import torch_snapshot
from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import Generator
from e2e_asr_th import Discriminator
from e2e_asr_th import pad_list

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 10


class CustomEvaluator(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x, aug=False)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, mtlalpha_decay=1.0):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.decay = mtlalpha_decay

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.ngpu > 1:
            loss = 1. / self.ngpu * self.model(*x)
            loss.backward(loss.new_ones(self.ngpu))  # Backprop
        else:
            loss = self.model(*x)
            loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
            self.model.mtlalpha *= self.decay

# Controls GAN PSDA / MMDA
class GANUpdater(training.StandardUpdater):
    '''
        Updater for GAN PSDA / MMDA
    '''
    def __init__(self, model, generator, discriminator, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, aug_params, gan_params, mtlalpha_decay=1.0):
        super(GANUpdater, self).__init__(train_iter, optimizer)
        
        # Class Attributes
        self.model = model
        self.generator = generator
        self.discriminator = discriminator
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.decay = mtlalpha_decay
        self.sources = train_iter.keys()

        # Augment Attributes
        self.aug_params = aug_params # Dict of all augmenting params
        self.done_augment = 0.0 # Counter for number of completed aug batches
        self.done_audio = 0.0   # Counter for number of completed audio batches
        self.done_pretrain_aug = 0.0 # Counter for pretrain aug batches completed
        self.gan_params = gan_params
        self.done_gan = 0.0
        self.done_disc = 0.0
   
    def _do_aug_batch(self):
        is_pretrain = (self.done_pretrain_aug < self.aug_params['pretrain'])
        self.done_pretrain_aug += is_pretrain 
        is_rand_aug = random.random() < self.aug_params['rat'] 
        return is_pretrain or is_rand_aug
    
    def update_core(self):
        if self.ngpu == 0:
            scale = 1.0
        else:
            scale = self.ngpu
        # Initialize GAN targets 
        loss_gan = 0.0
        if self.gan_params['weight'] > 0.0:
            ys_gan = torch.tensor((), dtype=torch.float32)
       
        # Augmenting Batch
        if self._do_aug_batch():
            self.done_augment += 1.0
            batch = self.get_iterator('aug').next()
            z = self.converter['aug'](batch, self.device)
            self.get_optimizer('aug').zero_grad()
            x_, ilens_ = self.generator(z[0], z[1])
            x = (x_, ilens_, z[2])
            is_aug = True

            # GAN 
            if self.gan_params['weight'] > 0.0 and x[0].shape[1] // 9 - 1 > self.gan_params['gan_wsize']:
                ys_gan = (ys_gan.new_ones((len(x[2]), 1))).to(self.device)
                loss_gan = (self.gan_params['weight'] / scale) * self.discriminator(x_, ilens_, ys_gan)  
                self.discriminator.reporter.report({'loss_gen': float(loss_gan)})
            
        # Audio Batch
        else: 
            batch = self.get_iterator('main').next()
            x = self.converter['main'](batch, self.device)
            self.done_audio += 1.0
            is_aug = False
                  
        # Compute the loss at this time step and accumulate it
        if self.model is not None:
            self.get_optimizer('main').zero_grad()  # Clear the parameter gradients
            loss = 1. / scale * self.model(*x, aug=is_aug) + loss_gan
            if self.ngpu > 1:
                loss.backward(loss.new_ones(self.ngpu))  # Backprop
            else:
                loss.backward()  # Backprop
            loss.detach()  # Truncate the graph

            # Update model parameters and generator parameters 
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
            logging.info('grad norm={}'.format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                self.get_optimizer('main').step()
                self.model.mtlalpha *= self.decay
 
            # Check if Generator is used
            if is_aug:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_threshold)
                logging.info('grad norm={}'.format(grad_norm))
                if math.isnan(grad_norm):
                    logging.warning('grad norm is nan. Do not update model.')
                else:
                    self.get_optimizer('aug').step()
        
        # If Discriminator is used
        if self.gan_params['weight'] > 0.0 and x[0].shape[1] // 9 - 1 > self.gan_params['gan_wsize']:
            # Discriminator Labels
            if is_aug:
                self.done_gan += 1.0
                print("GAN Accept Rate: ", self.done_gan / self.done_augment, self.done_gan, self.done_augment)
                label = self.gan_params['smooth'] * random.random()
            else:
                self.done_disc += 1.0
                print("Disc Accept Rate: ", self.done_disc / self.done_audio, self.done_disc, self.done_audio)
                label = 1.0 + self.gan_params['smooth'] * (random.random() - 1)
            ys_gan = (label * ys_gan.new_ones((len(x[2]), 1))).to(self.device)
            self.get_optimizer('gan').zero_grad()
            loss_discrim = (self.gan_params['weight'] / scale) * self.discriminator(x[0].detach(), x[1], ys_gan) 
            if self.ngpu > 1:
                loss_discrim.backward(loss.new_ones(self.ngpu))  # Backprop
            else:
                loss_discrim.backward()  # Backprop
            self.discriminator.reporter.report({'loss_discrim': float(loss_discrim)})
            loss_discrim.detach()  # Truncate the graph
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.grad_clip_threshold)
            logging.info('grad norm={}'.format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                self.get_optimizer('gan').step()


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

    def transform(self, item):
        return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


class AugmentConverter(object):
    """Augment CONVERTER"""

    def __init__(self, idict, odict, ifile, ofile, expand_iline=4, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1
        self.idict = idict
        self.odict = odict
        self.ifile = codecs.open(ifile, 'r', encoding='utf-8')
        self.ofile = codecs.open(ofile, 'r', encoding='utf-8')
        self.expand_iline = expand_iline

    def transform(self, item):
        return load_inputs_and_targets_augment(item, self.idict, self.odict,
                                 self.ifile, self.ofile, self.expand_iline)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x) for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    
    if args.tdnn_offsets != '':
        args.tdnn_offsets = [[int(o) for o in l.split(',')] for l in args.tdnn_offsets.split()]
    if args.tdnn_odims != '':
        args.tdnn_odims = [int(d) for d in args.tdnn_odims.split()]
    if len(args.tdnn_odims) != len(args.tdnn_offsets):
        sys.exit("Arguments are not right")

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    device = torch.device("cuda" if args.ngpu > 0 else "cpu")

    # specify model architecture
    if not args.gan_only:
        e2e = E2E(idim, odim, args)
        model = Loss(e2e, args.mtlalpha)
     
        # check the use of multi-gpu
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
            logging.info('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

        # set torch device
        model = model.to(device)
    
    if args.aug_use:
        gen = Generator(args.aug_vocab_size, args.aug_idim, idim, args)
        disc = Discriminator(idim, args.gan_odim, window_size=args.gan_wsize)
        gen = gen.to(device)
        disc = disc.to(device)

    # Setup an optimizer
    optimizer = {}
    if args.opt == 'adadelta':
        if not args.gan_only: 
            optimizer['main'] = torch.optim.Adadelta(
                model.parameters(), rho=0.95, eps=args.eps)
        if args.aug_use:
            optimizer['aug'] = torch.optim.Adadelta(gen.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        if not args.gan_only:
            optimizer['main'] = torch.optim.Adam(model.parameters())
        if args.aug_use:
            optimizer['aug'] = torch.optim.Adam(gen.parameters())

    if args.gan_weight > 0.0 and args.aug_use:
        optimizer['gan'] = torch.optim.SGD(disc.parameters(), lr=0.0008, momentum=0.5)
    
    # FIXME: TOO DIRTY HACK
    if not args.gan_only:
        setattr(optimizer['main'], "target", model.reporter)
        setattr(optimizer['main'], "serialize", lambda s: model.reporter.serialize(s))
    
    if args.aug_use and not args.gan_only:
        setattr(optimizer['aug'], "target", model.aug_reporter)
        setattr(optimizer['aug'], "serialize", lambda s: model.aug_reporter.serialize(s))
    elif args.aug_use:
        setattr(optimizer['aug'], "target", disc.reporter)
        setattr(optimizer['aug'], "serialize", lambda s: disc.reporter.serialize(s))

    if args.gan_weight > 0.0 and args.aug_use:
        setattr(optimizer['gan'], "target", disc.reporter)
        setattr(optimizer['gan'], "serialize", lambda s: disc.reporter.serialize(s))

    # Setup a converter
    #converter = CustomConverter(e2e.subsample[0])		
    converter = CustomConverter()

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

	# Augmenting Data
    if os.path.exists(args.aug_train) and args.aug_use:
        with codecs.open(args.aug_train, 'rb', encoding='utf-8') as f:
            augment_json = json.load(f)['aug']

        train_augment, meta = make_augment_batchset(augment_json, args.batch_size,
                                                    args.maxlen_in, args.maxlen_out,
                                                    args.minibatches, args.subsample)        
        
        if args.etype != 'tdnn':
            expand_iline = 4
        else:
            expand_iline = 3
        converter_augment = AugmentConverter(meta['idict'], meta['odict'],
                                             meta['ifilename'], meta['ofilename'],
                                             expand_iline=expand_iline,
                                             subsamping_factor=1) 


    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = {}
    if args.n_iter_processes > 0:
        train_iter['main'] = chainer.iterators.MultiprocessIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        if args.aug_use:
            train_iter['aug'] = chainer.iterators.MultiprocessIterator(
                TransformDataset(train_augment, converter_augment.transform),
                batch_size=1, n_process=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20) 
    else:
        train_iter['main'] = chainer.iterators.SerialIterator(
            TransformDataset(train, converter.transform),
            batch_size=1)
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False)
    if args.aug_use:
        train_iter['aug'] = chainer.iterators.SerialIterator(
            TransformDataset(train_augment, converter_augment.transform),
            batch_size=1)

    # Set up a trainer
    if args.aug_use:
        aug_params = {'rat': args.aug_ratio, 'pretrain': args.aug_pretrain} 
        gan_params = {'weight': args.gan_weight, 'smooth': args.gan_smooth, 'gan_wsize': args.gan_wsize}
        if args.gan_only:
            model = None
        updater = GANUpdater(model, gen, disc, args.grad_clip, train_iter,
                 optimizer, {'main': converter, 'aug': converter_augment}, device, args.ngpu, aug_params, gan_params, mtlalpha_decay=args.mtlalpha_decay)   
    else:
        updater = CustomUpdater(
            model, args.grad_clip, train_iter, optimizer, converter, device, args.ngpu, mtlalpha_decay=args.mtlalpha_decay)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if not args.gan_only:
        trainer.extend(CustomEvaluator(model, valid_iter, model.reporter, converter, device))

        # Save attention weight each epoch
        if args.num_save_attention > 0 and args.mtlalpha != 1.0:
            data = sorted(list(valid_json.items())[:args.num_save_attention],
                          key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
            if hasattr(model, "module"):
                att_vis_fn = model.module.predictor.calculate_all_attentions
            else:
                att_vis_fn = model.predictor.calculate_all_attentions
            trainer.extend(PlotAttentionReport(
                att_vis_fn, data, args.outdir + "/att_ws",
                converter=converter, device=device), trigger=(1, 'epoch'))

        # Make a plot for training and validation values
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                              'main/loss_ctc', 'validation/main/loss_ctc',
                                              'main/loss_att', 'validation/main/loss_att'],
                                             'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                             'epoch', file_name='acc.png'))

        # Save best models
        trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                       trigger=training.triggers.MinValueTrigger('validation/main/loss'))
        
        trainer.extend(extensions.snapshot_object(model, 'model.loss.best_{.updater.epoch}', savefun=torch_save),
                       trigger=training.triggers.MinValueTrigger('validation/main/loss'))

        if mtl_mode is not 'ctc':
            trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

            trainer.extend(extensions.snapshot_object(model, 'model.acc.best_{.updater.epoch}', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

   
        # epsilon decay in the optimizer
        if args.opt == 'adadelta':
            if args.criterion == 'acc' and mtl_mode is not 'ctc':
                trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            elif args.criterion == 'loss':
                trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    
    if args.aug_use:
        trainer.extend(extensions.snapshot_object(gen, 'gen.loss.best_{.updater.epoch}', savefun=torch_save),
                    trigger=(1, 'epoch'))
   
    # save snapshot which contains model and optimizer states
    #trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'aug/loss_ctc', 'aug/loss_att', 'aug/loss', 'aug/acc', 'gan/loss_discrim', 'gan/loss_gen',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if not args.gan_only:
        if args.opt == 'adadelta':
            trainer.extend(extensions.observe_value(
                'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
                trigger=(REPORT_INTERVAL, 'iteration'))
            report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=4*REPORT_INTERVAL))

    # Run the training
    trainer.run()
    if args.aug_use:
        converter_augment.ifile.close()
        converter_augment.ofile.close()

def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    torch_load(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
