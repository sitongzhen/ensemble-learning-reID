from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel
from scipy import io
import time
import scipy.io as scio
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from package.dataset import create_dataset
from package.model.Model import Model as Model

from package.utils.utils import time_str
from package.utils.utils import str2bool
from package.utils.utils import may_set_mode
from package.utils.utils import load_state_dict
from package.utils.utils import load_ckpt
from package.utils.utils import save_ckpt
from package.utils.utils import set_devices
from package.utils.utils import AverageMeter
from package.utils.utils import to_scalar
from package.utils.utils import ReDirectSTD
from package.utils.utils import set_seed
from package.utils.utils import adjust_lr_staircase

def may_make_dir(path):
  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def load0_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  ckpt = torch.load(ckpt_file, map_location=map_location)
  for m, sd in zip(modules_optims, ckpt['state_dicts']):
    m.load_state_dict(sd)
  if verbose:
    print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
      ckpt_file, ckpt['ep'], ckpt['scores']))
  return ckpt['ep'], ckpt['scores']


def save0_ckpt(modules_optims, ep, scores, ckpt_file):
  state_dicts = [m.state_dict() for m in modules_optims]
  ckpt = dict(state_dicts=state_dicts,
              ep=ep,
              scores=scores)
  may_make_dir(osp.dirname(osp.abspath(ckpt_file)))
  torch.save(ckpt, ckpt_file)

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(384, 128))
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1)

    parser.add_argument('--last_conv_stride', type=int, default=1,
                        choices=[1, 2])
    parser.add_argument('--num_stripes', type=int, default=1)
    parser.add_argument('--sample_ratio', type=int, default=0.8)
    parser.add_argument('--num_model', type=int, default=5)
    parser.add_argument('--local_conv_out_channels', type=int, default=256)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--new_params_lr', type=float, default=0.1)
    # parser.add_argument('--new_params_lr', type=float, default=0.1)
    parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(41,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=61)
    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None
    self.run = args.run

    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]
    self.ratio = args.sample_ratio
    self.num_model = args.num_model

    self.train_mirror_type = 'random' if args.mirror else None
    self.train_batch_size = args.batch_size
    self.train_final_batch = True
    self.train_shuffle = True

    self.test_mirror_type = None
    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    self.last_conv_stride = args.last_conv_stride
    self.num_stripes = args.num_stripes
    self.local_conv_out_channels = args.local_conv_out_channels

    self.momentum = 0.9
    self.weight_decay = 0.0005

    # Initial learning rate
    self.new_params_lr = args.new_params_lr
    self.finetuned_params_lr = args.finetuned_params_lr
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val
    self.steps_per_log = args.steps_per_log
    self.only_test = args.only_test
    self.resume = args.resume
    self.log_to_file = args.log_to_file

    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    self.ckpt_file = []
    for i in range(args.num_model):
        self.ckpt_file.append(osp.join(self.exp_dir, 'ckpt{:02d}.pth'.format(i)))

    self.model_weight_file = args.model_weight_file

def normalize_0(data):
    max = data.max(axis=1)
    min = data.min(axis=1)
    shape = data.shape
    data_row = shape[0]
    data_col = shape[1]
    t = np.empty((data_row, data_col))
    for i in range(data_col):
      t[:, i] = abs((data[:, i] - min) / (max - min))
    return t
def normalize(nparray, order=3, axis=1):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)
def softmax(data):
  x = np.exp(abs(data))
  for ss in range(x.size):
    qq = sum(x, 1)
    x[0, ss] = x[0, ss] / qq
  return x

class ExtractFeature(object):
  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = []
    for i in range(len(self.model)):
       old_train_eval_model.append(self.model[i].training)
       self.model[i].eval()

    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    try:
      local_feat_list = [], logits_list = []
      for i in range(cfg.num_model):
          local_feat_list_0, logits_list_0 = self.model[i](ims)
          local_feat_list.append(local_feat_list_0)
          logits_list.append(logits_list_0)

    except:
      local_feat_list = []
      for i in range(len(self.model)):
          local_feat_list_0 = self.model[i](ims)[0]
          local_feat_list.append(local_feat_list_0)
    feat_total = []
    for i in range(len(self.model)):
        feat = [lf.data.cpu().numpy() for lf in local_feat_list[i]]
        feat = np.concatenate(feat, axis=1)
        feat_total.append(feat)
    feat = feat_total[0]
    for i in range(len(self.model) - 1):
        feat += feat_total[i + 1]
    ########### with the integration weights ##############
    # wight_path = '/home/stz/PG_BOW_DEMO-master/w/1501-11743-rbf/'
    # for pp in range(256):
    #   num = pp + 1
    #   w_path = osp.join(wight_path + str(num) + 'w.mat')
    #   para_w = scio.loadmat(w_path)['w1']
    #   ww = abs(normalize(para_w))
    #   ww = torch.from_numpy(ww)
    #   ww_value = []
    #   new_feature = []
    #   for j in range(len(self.model)):
    #       ww_0 = ww[0, j];
    #       ww_value.append(ww_0)
    #
    #       new_feature.append(feat_total[j][:, pp] * ww_value[j])
    #
    # feat = new_feature[0]
    # for j in range(len(self.model) - 1):
    #     feat += new_feature[j+1]
    ##################################################
    # Restore the model to its old train/eval mode.
    for i in range(len(self.model)):
        self.model[i].train(old_train_eval_model[i])
    return feat#/len(self.model)


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  writer = None
  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  train_set = create_dataset(cfg.ratio, **cfg.train_set_kwargs)
  num_classes = len(train_set.ids2labels)
  val_set = create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  models = []
  for i in range(cfg.num_model):
      model = Model(last_conv_stride=cfg.last_conv_stride,num_stripes=cfg.num_stripes,
                  local_conv_out_channels=cfg.local_conv_out_channels,num_classes=num_classes)
      models.append(model)
  # Model wrapper
  model_w = []
  for i in range(cfg.num_model):
      model_w.append(DataParallel(models[i]))

  criterion = torch.nn.CrossEntropyLoss()
  ###################################################

  finetuned_params = []
  new_params = []
  param_groups = []
  optimizer = []
  modules_optims = []
  for i in range(cfg.num_model):
      finetuned_params.append(list(models[i].base.parameters()))
      new_params.append([p for n, p in models[i].named_parameters()
                if not n.startswith('base.')])
      param_groups.append([{'params': finetuned_params[i], 'lr': cfg.finetuned_params_lr},
                  {'params': new_params[i], 'lr': cfg.new_params_lr}])
      optimizer.append(optim.SGD(
                       param_groups[i],
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay))
      modules_optims.append([models[i], optimizer[i]])

  ##################################################
  if cfg.resume:
    for i in range(cfg.num_model):
        resume_ep, scores = load0_ckpt(modules_optims[i], cfg.ckpt_file[i])

  for i in range(cfg.num_model):
     TMO(modules_optims[i])


  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        for i in range(cfg.num_model):
            load0_ckpt(modules_optims[i], cfg.ckpt_file[i])

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=True,
        verbose=True)

  def validate():
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n===== Test on validation set =====\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=True,
      to_re_rank=False,
      verbose=True)
    print()
    return mAP, cmc_scores[0]

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):
    ###########################################
    loss_meter = []
    for num in range(cfg.num_model):
        adjust_lr_staircase(
           optimizer[num].param_groups,
           [cfg.finetuned_params_lr, cfg.new_params_lr],
           ep + 1,
           cfg.staircase_decay_at_epochs,
           cfg.staircase_decay_multiply_factor)

        loss_meter.append(AverageMeter())

        may_set_mode(modules_optims[num], 'train')

    ##########################################
    for num in range(cfg.num_model):
        features = []
        label = []

        ep_st = time.time()
        step = 0
        epoch0_done = False
        while not epoch0_done:
           step += 1
           step_st = time.time()

           ims, im_names, labels, mirrored, epoch0_done = train_set.next0_batch()

           ims_var = Variable(TVT(torch.from_numpy(ims).float()))
           labels_var = Variable(TVT(torch.from_numpy(labels).long()))

           local_fetures, logits_list = model_w[num](ims_var)

           feat = [lf.data.cpu().numpy() for lf in local_fetures]
           feat = np.concatenate(feat, axis=1)
           features.append(feat)
           label.append(labels)

           loss_0 = 0
           for i in range(len(logits_list)):
               loss_0 += criterion(logits_list[i], labels_var)
           optimizer[num].zero_grad()
           loss_0.backward()
           optimizer[num].step()

           loss_meter[num].update(to_scalar(loss_0))
           if step % cfg.steps_per_log == 0:
              log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
                     step, ep + 1, time.time() - step_st, loss_meter[num].val)
              print(log)

        ############ saving the features and labels ###################
        # features = np.vstack(features)
        # label = np.hstack(label)
        # dex = np.argsort(label)
        # label = label[dex]
        # features = features[dex]
        # io.savemat('/home/stz/PG_BOW_DEMO-master/duke-15000/training_feature{:2d}.mat'.format(i), {'features': features})
        # io.savemat('/home/stz/PG_BOW_DEMO-master/duke-15000/training_label{:2d}.mat'.format(i), {'labels': label})

        log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
               ep + 1, time.time() - ep_st, loss_meter[num].avg)
        print(log)

        save0_ckpt(modules_optims[num], ep + 1, 0, cfg.ckpt_file[num])

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    if (ep + 1) % cfg.epochs_per_val == 0:
      mAP, Rank1 = validate()

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
        ep)
      writer.add_scalars(
        'loss',
         dict(loss=loss_meter[0].avg, ),
        ep)

  ########
  # Test #
  ########
  test(load_model_weight=False)

if __name__ == '__main__':
  main()
