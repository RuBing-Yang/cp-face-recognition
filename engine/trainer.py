import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.meter import Meter, AverageMeter, ProgressMeter


def save(model, ckpt_num, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    if torch.cuda.device_count() > 1:
        torch.save(
            model.module.state_dict(), 
            os.path.join(dir_name, 'model_%s.pth.tar' % ckpt_num)
        )
    else:
        torch.save(
            model.state_dict(), 
            os.path.join(dir_name, 'model_%s.pth.tar' % ckpt_num)
        )
    print('model saved!')


def fit(train_loader, nb_epoch,  
        cartoon_encoder, face_encoder, 
        loss_fn, optimizer, scheduler, 
        device, log_interval, save_interval, 
        start_epoch=0, save_model_to='./tmp/save_model_to'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    # Save pre-trained model
    save(cartoon_encoder, 0, save_model_to)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, nb_epoch):
        

        # Train stage
        train_loss = train_epoch(train_loader, cartoon_encoder, face_encoder, 
                                 loss_fn, optimizer, device, log_interval)

        log_dict = {'epoch': epoch + 1,
                    'epoch_total': nb_epoch,
                    'loss': float(train_loss),
                    }

        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, nb_epoch, train_loss)
 
        print(message)
        print(log_dict)
        if (epoch + 1) % save_interval == 0:
            save(cartoon_encoder, epoch + 1, save_model_to)


def class_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def retrieval_accuracy(sims, flags, thres_step=1e-2):
    assert(thres_step > 0 and thres_step < 1)
    thres_range = np.arange(-1.0, 1.0, thres_step)

    successes = torch.Tensor([
        torch.sum((sims > thres) == flags)
        for thres in thres_range
    ])
    
    best_idx = torch.argmax(successes)
    best_thres = thres_range[best_idx]
    acc = successes[best_idx] / len(flags)

    return best_thres, acc 


def train_epoch(train_loader, cartoon_encoder, face_encoder, 
                loss_fn, optimizer, device, log_interval):
    cartoon_encoder.train()
    total_loss = 0

    for batch_idx, (faces, cartoons, targets) in enumerate(train_loader):
        targets = targets if len(targets) > 0 else None

        faces = faces.to(device)
        cartoons = cartoons.to(device)

        if targets is not None:
            targets = targets.to(device)

        optimizer.zero_grad()
        if loss_fn.cross_entropy_flag:
            # ! deprecated
            assert(False) 
            # output_embedding, output_cross_entropy = cartoon_encoder(*data)
            # blended_loss, losses = loss_fn.calculate_loss(target, output_embedding, output_cross_entropy)
        else:
            # NOTE: embedding
            #   1. encode cartoons with cartoon_encoder 
            #   2. encode faces with face_encoder 
            #   3. calculate the blended_loss(n-pair loss + angular oss) with
            #       - anchors=cartoons
            #       - positives=face pictures of the same person as anchors
            #       - negatives= other peoples' face pictures
            cartoon_embeddings = cartoon_encoder(cartoons)
            faces_embeddings = face_encoder(faces)

            assert cartoon_embeddings.shape == faces_embeddings.shape, \
                   "both picture need to be projected into same space"
            
            output_embeddings = torch.cat([cartoon_embeddings, faces_embeddings], dim=0)
            
            blended_loss, losses = loss_fn.calculate_loss(
                targets, output_embeddings)
        total_loss += blended_loss.item()
        blended_loss.backward()

        optimizer.step()

        # Print log
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(faces[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader)) # FIXME: progress counting error
            for name, value in losses.items():
                message += '\t{}: {:.6f}'.format(name, np.mean(value))
 
            print(message)

    total_loss /= (batch_idx + 1)
    return total_loss


def train(train_loader, net, metric, criterion, 
          optimizer, lr_scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':5.3f')
    data_time = AverageMeter('Data', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.02f')
    top5 = AverageMeter('Acc@5', ':6.02f')
    learning_rate = Meter('learnig rate', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, learning_rate, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    net.train()
    metric.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        raw_logits = net(images)
        output = metric(raw_logits, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = class_accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        learning_rate.update(lr_scheduler.get_lr()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, val_dataset_size, 
             net, epoch, args, 
             visualizer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, ], prefix='Test: '
    )

    # switch to evaluate mode
    net.eval()

    pairs = [(i, j) 
        for i in range(val_dataset_size) 
        for j in range(val_dataset_size)
        if i < j
    ]

    feats = []
    # fliped_feats = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            
            target = target.cuda(args.gpu, non_blocking=True)
            
            # fliped_images = images.flip(-1) 
            
            # compute output
            feat = net(images)
            # fliped_feat = net(fliped_images)

            feats.append(feat)
            # fliped_feats.append(fliped_feat)
            targets.append(target) 

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.display(i + 1)        

    feats = torch.cat(feats, dim=0)
    # fliped_feats = torch.cat(fliped_feats, dim=0)
    targets = torch.cat(targets, dim=0) 

    flags = torch.Tensor([ 
        targets[i] == targets[j] 
        for (i, j) in pairs
    ]).cuda().type(torch.bool)

    assert(len(pairs) == len(flags))
    sims = torch.cat([
        F.cosine_similarity(feats[i: i+1], feats[j: j+1])
        for (i, j) in pairs
    ]).cuda()

    # measure accuracy and record loss
    thres, acc = retrieval_accuracy(sims, flags, thres_step=5e-3)

    print(f'Epoch: [{epoch}] * Threshold {thres:6.3f}\tAcc {acc*100:4.2f}')

    return thres, acc