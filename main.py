# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# from posix import PRIO_USER
import shutil
import argparse
import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

import torchvision
from torchvision import transforms

from utils.augmentation import RandAugment
from utils.datasets import BalancedBatchSampler
from utils.dataloader import train_data_loader, test_data_loader

# Load initial models
from architecture.networks import EmbeddingNetwork
from architecture.metric import ArcMarginProduct

# Load batch sampler and train loss
from losses import BlendedLoss, MAIN_LOSS_CHOICES

from engine.trainer import fit, train, validate
from engine.inference import retrieve # TODO: implement inference.py

from facenet_pytorch import InceptionResnetV1

def save_checkpoint(state, is_best, save_dir='', filename='checkpoint.pth.tar'):
    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(
            file_path, 
            os.path.join(save_dir, 'model_best.pth.tar')
        )


def load(model, file_path):
    model.load_state_dict(torch.load(file_path))
    print('model loaded!')
    return model


def infer(input_size, face_encoder, cartoon_encoder,  queries, db):
    retrieval_results = retrieve(face_encoder, cartoon_encoder, queries, db, input_size)

    return list(zip(range(len(retrieval_results)), retrieval_results.items()))


def get_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset-path', type=str)
    args.add_argument('--save-dir', type=str, default='./model')
    args.add_argument('--face-encoder', type=str, 
                      help='checkpoint path of face encoder.')   # deprecated
    args.add_argument('--cartoon-encoder', type=str,
                      help='checkpoint path of cartoon encoder.')
    args.add_argument('--workers', type=int, default=4)
    args.add_argument('--gpu', type=int, default=0)

    # Hyperparameters
    args.add_argument('--start-epoch', type=int, default=0)
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--model', type=str,
                      choices=['densenet161', 'resnet101',  'inceptionv3', 'seresnext'],
                      default='seresnext')
    args.add_argument('--input-size', type=int, default=112, help='size of input image')
    args.add_argument('--num-classes', type=int, default=64, help='number of classes for batch sampler')
    args.add_argument('--num-samples', type=int, default=1, help='number of samples per class for batch sampler')
    args.add_argument('--embedding-dim', type=int, default=512, help='size of embedding dimension')
    args.add_argument('--feature-extracting', type=bool, default=False)
    args.add_argument('--use-pretrained', action='store_true', default=True)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--scheduler', type=str, choices=['StepLR', 'MultiStepLR'])
    args.add_argument('--attention', action='store_true')
    args.add_argument('--loss-type', type=str, default='n-pair-angular', choices=MAIN_LOSS_CHOICES)
    args.add_argument('--cross-entropy', action='store_true')
    args.add_argument('--use-augmentation', action='store_true')

    args.add_argument('-s', '--scale', type=float, default=32, 
                      help='scale s in Arcface.')
    args.add_argument('-m', '--margin', type=float, default=0.5,
                      help='margin m in Arcface.')
    
    # Mode selection
    args.add_argument('--mode', type=str, choices=['train-face', 'train-all', 'test'], 
                      required=True, help='mode selection')
    args.add_argument('--log-interval', type=int, default=50)
    args.add_argument('--save-interval', type=int, default=20)
    
    return args.parse_args()


best_acc1 = 0

def main(config):
    global best_acc1

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # if ((config.mode == 'train-all' or 
    #      config.mode == 'test') and 
    #      config.face_encoder is None):
    #     raise ValueError('Trained face encoder is required.')

    dataset_path = config.dataset_path

    # Model parameters
    model_name = config.model
    workers = config.workers
    input_size = config.input_size
    embedding_dim = config.embedding_dim
    feature_extracting = config.feature_extracting
    use_pretrained = config.use_pretrained
    attention_flag = config.attention

    # Training parameters
    start_epoch = config.start_epoch
    nb_epoch = config.epochs
    loss_type = config.loss_type
    cross_entropy_flag = False # config.cross_entropy
    scheduler_name = config.scheduler
    lr = config.lr

    # Mini-batch parameters
    num_classes = config.num_classes
    num_samples = config.num_samples
    use_augmentation = config.use_augmentation

    infer_batch_size = 64
    log_interval = config.log_interval
    save_interval = config.save_interval
    start_epoch = 0

    if config.mode == 'train-face':
        assert(False)

         # create model
        print("=> creating backbone '{}'".format(model_name))
        net = EmbeddingNetwork(model_name=model_name,
                                 embedding_dim=embedding_dim,
                                 feature_extracting=feature_extracting,
                                 use_pretrained=use_pretrained,
                                 attention_flag=attention_flag,
                                 cross_entropy_flag=cross_entropy_flag)

        # create margin
        print("=> creatinig margin '{}'".format(config.margin))
        metric = ArcMarginProduct(in_feature=embedding_dim, 
                                  out_feature=num_classes, 
                                  s=config.scale,
                                  m=config.margin)

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            metric = nn.DataParallel(metric)

        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            net = net.cuda(config.gpu)
            metric = metric.cuda(config.gpu)
        else:
            net = torch.nn.DataParallel(net).cuda()
            metric = torch.nn.DataParallel(metric).cuda()
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(metric.gpu)

        optimizer = torch.optim.SGD([
            {'params': net.parameters(), 'weight_decay': 5e-4},
            {'params': metric.parameters(), 'weight_decay': 5e-4}
        ], lr, momentum=config.momentum, nesterov=True)
        
        exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[6, 11, 16], gamma=0.1
        )

        # optionally resume from a checkpoint
        if config.resume_face is not None:
            if os.path.isfile(config.resume):
                print("=> loading checkpoint '{}'".format(config.resume_face))
                checkpoint = torch.load(config.resume_face)
                assert(checkpoint['type'] == 'face_encoder')
                start_epoch = checkpoint['epoch']
                
                net.load_state_dict(checkpoint['net_state_dict'])
                metric.load_state_dict(checkpoint['metric_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume_face, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(config.resume_face))

        cudnn.benchmark = True

        # Data loading code
        traindir = os.path.join(dataset_path, 'train')  
        valdir = os.path.join(dataset_path, 'val')
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        is_valid_file = lambda path: \
            os.path.basename(path)[0] == 'P'    # only use human faces for training 

        original_train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(128),
                transforms.RandomResizedCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), is_valid_file=is_valid_file)

        augmented_train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(128),
                transforms.RandomResizedCrop(112),
                RandAugment(n=2, m=10),
                transforms.ToTensor(),
                normalize,
            ]), is_valid_file=is_valid_file)

        train_dataset = torch.utils.data.ConcatDataset([
            original_train_dataset, augmented_train_dataset
        ])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=num_samples, shuffle=True,
            num_workers=config.workers, pin_memory=True
        )

        # FIXME: centrecrop may corp some useful info 
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=num_samples, shuffle=False,
            num_workers=config.workers, pin_memory=True)


        for _ in range(start_epoch):
            exp_lr_scheduler.step()

        for epoch in range(start_epoch, nb_epoch):
            # train for one epoch
            train(train_loader, net, metric, criterion, 
                optimizer, exp_lr_scheduler, epoch, config)

            # adjust_learning_rate
            exp_lr_scheduler.step()

            # evaluate on validation set
            _, acc1 = validate(val_loader, net, epoch, config)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if epoch % config.save_freq == 0:
                filename="checkpoint-{:d}.pth.tar".format(epoch)
                path = os.path.join(config.save_dir, filename)
                save_checkpoint({
                    'type'              :   'face_encoder',
                    'epoch'             :   epoch + 1,
                    'backbone'          :   config.backbone,
                    'metirc'            :   config.margin,
                    'net_state_dict'    :   net.state_dict(),
                    'margin_state_dict' :   metric.state_dict(),
                    'optimizer'         :   optimizer.state_dict(),
                }, is_best, path)
        
        return


    """ Model """
    # load facenet as face-encoder 
    face_encoder = InceptionResnetV1(pretrained='vggface2').eval()  # TODO: replace with iresnet100-arcface later 
    cartoon_encoder = EmbeddingNetwork(model_name=model_name,
                                       embedding_dim=embedding_dim,
                                       feature_extracting=feature_extracting,
                                       use_pretrained=use_pretrained,
                                       attention_flag=attention_flag,
                                       cross_entropy_flag=cross_entropy_flag)

    if config.cartoon_encoder is not None:
        load(cartoon_encoder, file_path=config.cartoon_encoder)

    if torch.cuda.device_count() > 1: 
        face_encoder = nn.DataParallel(face_encoder).eval()
        cartoon_encoder = nn.DataParallel(cartoon_encoder)

    if config.mode == 'train-all':

        """ Load data """
        train_dataset_path = os.path.join(dataset_path, 'train')
        
        img_dataset = train_data_loader(data_path=train_dataset_path, img_size=input_size,
                                        use_augment=use_augmentation)
        # NOTE: dataloading
        #   yield a batch of data as following
        #   size: (N, channel, height, width)        
        #   anchor(face image) * 1, positive(cartoon images) * 1, negatives(cartoon images) * (N - 2)
        #       Balanced batch sampler and online train loader
        train_batch_sampler = BalancedBatchSampler(img_dataset, n_classes=num_classes, n_samples=num_samples)
        online_train_loader = torch.utils.data.DataLoader(img_dataset,
                                                          batch_sampler=train_batch_sampler,
                                                          num_workers=workers,
                                                          pin_memory=True)

        device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

        # Gather the parameters to be optimized/updated.
        params_to_update = cartoon_encoder.parameters()
        print("Params to learn:")
        if feature_extracting:
            params_to_update = []   
            for name, param in cartoon_encoder.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in cartoon_encoder.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Send the model to GPU
        face_encoder = face_encoder.to(device)
        cartoon_encoder = cartoon_encoder.to(device)

        optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        elif scheduler_name == 'MultiStepLR':
            # TODO: tuing the lr decay for better optimization
            if use_augmentation:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[20, 30], gamma=0.1
                )
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[10, 15, 20], gamma=0.1
                )
        else:
            raise ValueError('Invalid scheduler')

        # Loss function
        loss_fn = BlendedLoss(loss_type, cross_entropy_flag)

        # Train (fine-tune) model
        fit(online_train_loader, nb_epoch,
            cartoon_encoder, face_encoder, 
            loss_fn, optimizer, scheduler, 
            device, log_interval, save_interval,
            start_epoch=start_epoch,
            save_model_to=config.save_dir)
    
    elif config.mode == 'test':
        # TODO: not implemented yet
        test_dataset_path = os.path.join(dataset_path + '/test')
        queries, db = test_data_loader(test_dataset_path)
        cartoon_encoder = load(cartoon_encoder, file_path=config.cartoon_encoder)
        result = infer(input_size, face_encoder, cartoon_encoder, queries, db)
        result_dict = {}
        with open('result.json', 'r') as f:
            result = json.load(f)
            for i in result:
                key = i[1][0].split('\\')[-1]
                val = i[1][1][0].split('\\')[-1]
                result_dict[key] = val.strip()

        pwd = os.path.dirname(os.path.abspath(__file__))
        ans = []

        with open(pwd + '\FR_Probe_C2P.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                ans.append(result_dict[line[0]])

        with open('result.csv', 'w') as f:
            for a in ans:
                f.write(a + '\n')


        # TODO: save inference result
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(config=get_arguments())
    