# -*- coding: utf-8 -*-
# @Time    : 2023/11/14
# @Author  : White Jiang
# -*- coding: utf-8 -*-
# @Time    : 2023/11/4
# @Author  : White Jiang
import logging
import dataset
import utils

import os

import torch
import numpy as np
import time
import argparse
import json
import random
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from model.vit import VisionTransformer, CONFIGS
from loguru import logger
import datetime
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Training ProxyNCA++')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config.json')
parser.add_argument('--embedding-size', default=2048, type=int, dest='sz_embedding')
parser.add_argument('--batch-size', default=32, type=int, dest='sz_batch')
parser.add_argument('--epochs', default=40, type=int, dest='nb_epochs')
parser.add_argument('--log-filename', default='example')
parser.add_argument('--workers', default=16, type=int, dest='nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'trainval', 'test'],
                    help='train with train data or train with trainval')
parser.add_argument('--lr_steps', default=[100], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--eval_nmi', default=False, action='store_true')
parser.add_argument('--recall', default=[1, 2, 4, 8], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--no_warmup', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--output_dir', default='/home/s02007/code/proxynca_pp-master/logs', type=str)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('log'):
    os.makedirs('log')

output_dir = args.output_dir + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger.add(output_dir + '/' + 'train_{}.log'.format(args.dataset))
logger.info("Saving model in the path: {}".format(output_dir))

curr_fn = os.path.basename(args.config).split(".")[0]

out_results_fn = "log/%s_%s_%s_%d.json" % (args.dataset, curr_fn, args.mode, args.seed)

config = utils.load_config(args.config)

dataset_config = utils.load_config('dataset/config.json')

if args.source_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
    dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
if args.root_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
    dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

# set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
if args.mode == 'trainval' or args.mode == 'test':
    args.eval_nmi = True

args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
args.sz_embedding = config['sz_embedding']
if 'warmup_k' in config:
    args.warmup_k = config['warmup_k']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']

args.log_filename = '%s_%s_%s_%d' % (args.dataset, curr_fn, args.mode, args.seed)
if args.mode == 'test':
    args.log_filename = args.log_filename.replace('test', 'trainval')

best_epoch = args.nb_epochs

config_model = CONFIGS['ViT-B_16']
feat = VisionTransformer(config_model, 224)
feat.load_from(np.load('/home/s02007/code/FG_CLS/checkpoint/ViT-B_16.npz'))
feat.eval()
feat.train()

model = feat.cuda()

model = torch.nn.DataParallel(model)


def save_best_checkpoint(model):
    torch.save(model.state_dict(), 'results/' + args.log_filename + '.pt')


def load_best_checkpoint(model):
    model.load_state_dict(torch.load('results/' + args.log_filename + '.pt'))
    model = model.cuda()
    return model


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.3
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2, activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


if args.mode == 'trainval':
    train_results_fn = "log/%s_%s_%s_%d.json" % (args.dataset, curr_fn, 'train', args.seed)
    if os.path.exists(train_results_fn):
        with open(train_results_fn, 'r') as f:
            train_results = json.load(f)
        args.lr_steps = train_results['lr_steps']
        best_epoch = train_results['best_epoch']


train_transform = transforms.Compose([
                                      transforms.Resize((256, 256), Image.BILINEAR),
                                      transforms.RandomCrop([224, 224]),
                                      transforms.RandomHorizontalFlip(),
                                      GaussianBlur(),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.ColorJitter(0.3, 0.3, 0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                     transforms.CenterCrop([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
print('best_epoch', best_epoch)

results = {}

dl_ev = torch.utils.data.DataLoader(
    dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
        transform=test_transform
    ),
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
)

if args.mode == 'train':
    tr_dataset = dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['train'],
        transform=train_transform
    )
elif args.mode == 'trainval' or args.mode == 'test':
    tr_dataset = dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
        transform=train_transform
    )

num_class_per_batch = config['num_class_per_batch']
num_gradcum = config['num_gradcum']
is_random_sampler = config['is_random_sampler']
if is_random_sampler:
    batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch,
                                                     num_gradcum)
else:
    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch,
                                                       int(args.sz_batch / num_class_per_batch))

dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler=batch_sampler,
    num_workers=args.nb_workers,
    # pin_memory = True
)

print("===")
if args.mode == 'train':
    dl_val = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['val'],
            transform=test_transform
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # drop_last=True
        # pin_memory = True
    )

criterion = config['criterion']['type'](
    nb_classes=dl_tr.dataset.nb_classes(),
    sz_embed=args.sz_embedding,
    **config['criterion']['args']
).cuda()

opt_warmup = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            'lr': 0
        },
        {
            **{'params': criterion.parameters()}
            ,
            **config['opt']['args']['proxynca']

        },

    ],
    **config['opt']['args']['base']
)

opt = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            **config['opt']['args']['backbone']
        },
        {
            **{'params': criterion.parameters()},
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base']
)
t_total = len(dl_tr) * args.nb_epochs
if args.mode == 'test':
    with torch.no_grad():
        logger.info("**Evaluating...(test mode)**")
        model = load_best_checkpoint(model)
        utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)

    exit()

if args.mode == 'train':
    scheduler = config['lr_scheduler']['type'](
        opt, **config['lr_scheduler']['args']
    )
elif args.mode == 'trainval':
    scheduler = config['lr_scheduler2']['type'](
        opt,
        milestones=args.lr_steps,
        gamma=0.1
    )
    # scheduler = WarmupCosineSchedule(opt, warmup_steps=500, t_total=t_total)

logger.info("Training parameters: {}".format(vars(args)))
logger.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()

if args.init_eval:
    logger.info("**Evaluating initial model...**")
    with torch.no_grad():
        if args.mode == 'train':
            c_dl = dl_val
        else:
            c_dl = dl_ev

        utils.evaluate(model, c_dl, args.eval_nmi, args.recall)  # dl_val

it = 0

best_val_nmi = 0
best_val_epoch = 0
best_val_r1 = 0
best_test_nmi = 0
best_test_r1 = 0
best_test_r2 = 0
best_test_r5 = 0
best_test_r8 = 0
best_tnmi = 0

prev_lr = opt.param_groups[0]['lr']
lr_steps = []

print(len(dl_tr))

if not args.no_warmup:
    # warm up training for 5 epochs
    logger.info("**warm up for %d epochs.**" % args.warmup_k)
    for e in range(0, args.warmup_k):
        for ct, (x, y, _) in enumerate(dl_tr):
            opt_warmup.zero_grad()
            m = model(x.cuda())
            cons_loss = con_loss(m, y.cuda())
            loss1 = criterion(m, y.cuda())
            loss = loss1 + cons_loss
            # loss = loss1

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()
        logger.info('warm up ends in %d epochs' % (args.warmup_k - e))

for e in range(0, args.nb_epochs):

    if args.mode == 'train':
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    tnmi = []
    opt.zero_grad()
    for ct, (x, y, _) in enumerate(dl_tr):
        it += 1

        m = model(x.cuda())

        loss1 = criterion(m, y.cuda())
        cons_loss = con_loss(m, y.cuda())
        loss = loss1 + cons_loss
        # loss = loss1

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())

        if (ct + 1) % 1 == 0:
            opt.step()
            opt.zero_grad()

    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    print('it: {}'.format(it))
    logger.info('Current lr: {}'.format(opt.param_groups[0]['lr']))
    logger.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )

    model.losses = losses
    model.current_epoch = e

    if e == best_epoch:
        break

    if args.mode == 'trainval':
        scheduler.step(e)
        with torch.no_grad():
            logging.info("**Validation...**")
            nmi, recall = utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)
        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])
        logger.info('Current val nmi: {}'.format(nmi))
        logger.info('Current val r1: {}'.format(recall[0]))
        logger.info('Current val r2: {}'.format(recall[1]))
        logger.info('Current val r4: {}'.format(recall[2]))
        logger.info('Current val r8: {}'.format(recall[3]))
        if recall[0] > best_val_r1:
            best_val_nmi = nmi
            best_val_r1 = recall[0]
            best_val_r2 = recall[1]
            best_val_r4 = recall[2]
            best_val_r8 = recall[3]
            best_val_epoch = e
            best_tnmi = torch.Tensor(tnmi).mean()
        if e == (args.nb_epochs - 1):
            # saving best epoch
            results['best_NMI'] = best_val_nmi
            results['best_R1'] = best_val_r1
            results['best_R2'] = best_val_r2
            results['best_R4'] = best_val_r4
            results['best_R8'] = best_val_r8
            logger.info('Best val epoch: {}'.format(best_val_epoch))
            logger.info('Best val nmi: {}'.format(best_val_nmi))
            logger.info('Best val r1: {}'.format(best_val_r1))
            logger.info('Best val r2: {}'.format(best_val_r2))
            logger.info('Best val r4: {}'.format(best_val_r4))
            logger.info('Best val r8: {}'.format(best_val_r8))

# if args.mode == 'trainval':
#     save_best_checkpoint(model)
    torch.save(model.state_dict(), 'results/' + 'cub_{}.pth'.format(e))
    with torch.no_grad():
        logger.info("**Evaluating...**")
        model = load_best_checkpoint(model)
        best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8) = utils.evaluate(model, dl_ev,
                                                                                                 args.eval_nmi,
                                                                                                 args.recall)
        # logging.info('Best test r8: %s', str(best_test_r8))

    results['NMI'] = best_test_nmi
    results['R1'] = best_test_r1
    results['R2'] = best_test_r2
    results['R4'] = best_test_r4
    results['R8'] = best_test_r8

if args.mode == 'train':
    print('lr_steps', lr_steps)
    results['lr_steps'] = lr_steps

with open(out_results_fn, 'w') as outfile:
    json.dump(results, outfile)

t2 = time.time()
