import argparse
import os
import shutil
import time
import pickle
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# self-defined libs: imprinted
import models
import loader
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# self-defined libs: basemodel
# EPIC loader
from basemodel.utils.other import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='CUB_200_2011',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='pretrain_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# options added for EPIC
parser.add_argument('--dataset', default='EPIC',
                    help='dataset name')
# parser.add_argument('--foptions', default='/vision2/u/bingbin/ORN/ckpt/epic_headscontext+object+star_gcnObj0_gcnCtxt0_bt2_lr5e-05_wd1e-05_star_v2_visPrior_correctBBox/options.pkl',
parser.add_argument('--foptions', default='/vision2/u/bingbin/ORN/ckpt/epic_headscontext+object+star_gcnObjNone_gcnCtxtNone_bt4_lr1e-04_wd3e-06_tuneConv3D_adjWNorm_freezeORN_maxObj/options.pkl',
                    help='Path to an option file from a previous ckpt.')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print('dataset:', args.dataset)
    if args.dataset == 'EPIC':
      with open(args.foptions, 'rb') as handle:
        options = pickle.load(handle)
        if options['root'][0] == '.':
          # update relative path
          options['root'] = 'basemodel' + options['root'][1:]
        options['num_crops'] = 1
    else:
      options = None

    model = models.Net(options=options).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())

    optimizer = torch.optim.SGD([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 10}
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optionally resume from a checkpoint
    # title = 'CUB'
    title = args.dataset

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    if args.dataset == 'CUB':
      print('Using dataset CUB')
      train_dataset = loader.ImageLoader(
          args.data,
          transforms.Compose([
              transforms.RandomResizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize,
          ]), train=True)
  
      train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=args.batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True)
  
      val_loader = torch.utils.data.DataLoader(
          loader.ImageLoader(args.data, transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])),
          batch_size=args.batch_size, shuffle=False,
          num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'EPIC':
      print('Using dataset EPIC')
      train_dataset, val_dataset, train_loader, val_loader = get_datasets_and_dataloaders(options, device=options['device'])

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print('# train_loader:', len(train_loader))
    print('# val_loader:', len(val_loader))

    print('Training epoch from {} to {}'.format(args.start_epoch, args.epochs))

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        test_loss, test_acc = validate(val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))

    try:
      for batch_idx, input_var in enumerate(train_loader):
  
          if (type(input_var)==tuple or type(input_var)==list) and len(input_var) == 2:
            input, target = input_var
          elif type(input_var) == dict:
            input = input_var
            target = input_var['target']
            if target.dim() == 2:
              # prepare format for CrossEntropy
              target = target.nonzero()[:, 1]
          else:
            raise ValueError('Unknown type: type(input_var)=={}'.format(type(input_var)))
  
          # measure data loading time
          data_time.update(time.time() - end)
  
          if type(input) == dict:
            for key in input:
              input[key] = input[key].contiguous().cuda()
          else:
            input = input.cuda()
          target = target.cuda()
  
          # compute output
          output = model(input)
          loss = criterion(output, target)
  
          # measure accuracy and record loss
          prec1, prec5 = accuracy(output, target, topk=(1, 5))
  
          losses.update(loss.item(), target.size(0))
          top1.update(prec1.item(), target.size(0))
          top5.update(prec5.item(), target.size(0))
  
          # compute gradient and do SGD step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()
  
          model.weight_norm()
          # plot progress
          bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                      batch=batch_idx + 1,
                      size=len(train_loader),
                      data=data_time.val,
                      bt=batch_time.val,
                      total=bar.elapsed_td,
                      eta=bar.eta_td,
                      loss=losses.avg,
                      top1=top1.avg,
                      top5=top5.avg,
                      )
          bar.next()
    except Exception as e:
      bar.finish()
      raise e
    bar.finish()
    return (losses.avg, top1.avg)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing ', max=len(val_loader))
    try:
      with torch.no_grad():
          end = time.time()
          for batch_idx, input_var in enumerate(val_loader):
              if (type(input_var) == tuple or type(input_var) == list) and len(input_var) == 2:
                input, target = input_var
              elif type(input_var) == dict:
                input = input_var
                target = input_var['target']
                if target.dim() == 2:
                  # prepare format for CrossEntropy
                  target = target.nonzero()[:, 1]
              else:
                raise ValueError('Unknown type: type(input_var)=={}'.format(type(input_var)))

              # measure data loading time
              data_time.update(time.time() - end)

              if type(input) == dict:
                for key in input:
                  try:
                    input[key] = input[key].cuda()
                  except Exception as e:
                    print('key:', key)
                    print(e)
                    pdb.set_trace()
              else:
                input = input.cuda()
              target = target.cuda()

              # compute output
              output = model(input)
              loss = criterion(output, target)

              # measure accuracy and record loss
              prec1, prec5 = accuracy(output, target, topk=(1, 5))
              losses.update(loss.item(), target.size(0))
              top1.update(prec1.item(), target.size(0))
              top5.update(prec5.item(), target.size(0))

              # measure elapsed time
              batch_time.update(time.time() - end)
              end = time.time()

              # plot progress
              bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                          batch=batch_idx + 1,
                          size=len(val_loader),
                          data=data_time.avg,
                          bt=batch_time.avg,
                          total=bar.elapsed_td,
                          eta=bar.eta_td,
                          loss=losses.avg,
                          top1=top1.avg,
                          top5=top5.avg,
                          )
              bar.next()
          bar.finish()
    except Exception as e:
      bar.finish()
      raise e

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
