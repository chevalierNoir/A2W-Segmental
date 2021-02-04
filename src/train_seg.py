import os
import sys
import socket
import logging
import argparse
import torch
import lev
import utils
import time
import options
import yaml
import numpy as np
import torch.utils.data as tud
import torch.optim as optim
import delta_ark as dataset
from torch import nn
from seg_model import SegModel
from apex import amp


def train(encoder, optimizer, loader, ckpt, opt):
    encoder.train()
    larr, lseg_arr, lemb_arr = [], [], []
    accum_batch, loss_accum, loss_seg_accum, loss_emb_accum = 0, 0, 0, 0
    optimizer.zero_grad()
    logging.info(f"{len(loader)} training batches")
    for i_batch, sample in enumerate(loader):
        feats, labels, label_batch, prob_sizes, label_sizes = sample['feature'], sample['label'], sample['label_batch'], sample['prob_size'], sample['label_size']
        _, _, l, loss_seg, loss_emb = encoder(feats, label_sizes, prob_sizes, label_batch, emb_agwe=opt.emb_agwe)
        if opt.accum_batch_size > 0:
            num_accum_steps = max(opt.accum_batch_size // label_sizes.size(0), 1)
            l, loss_seg, loss_emb = l / num_accum_steps, loss_seg / num_accum_steps, loss_emb / num_accum_steps
            loss_accum += l
            loss_seg_accum += loss_seg
            loss_emb_accum += loss_emb
            if opt.amp == 1:
                with amp.scale_loss(l, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l.backward()
            accum_batch += label_sizes.size(0)
            if accum_batch >= opt.accum_batch_size:
                larr.append(loss_accum)
                lseg_arr.append(loss_seg_accum)
                lemb_arr.append(loss_emb_accum)
                accum_batch = 0
                loss_accum, loss_seg_accum, loss_emb_accum = 0, 0, 0
                if opt.amp == 1:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad)
                else:
                    torch.nn.utils.clip_grad_norm(encoder.parameters(), opt.max_grad)
                optimizer.step()
                optimizer.zero_grad()
        else:
            if opt.amp == 1:
                with amp.scale_loss(l, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l.backward()
                if opt.amp == 1:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad)
                else:
                    torch.nn.utils.clip_grad_norm(encoder.parameters(), opt.max_grad)
            optimizer.step()
            optimizer.zero_grad()
            larr.append(l.item())
            lseg_arr.append(loss_seg.item())
            lemb_arr.append(loss_emb.item())

        ckpt['step'] += 1
        if ckpt['step'] % opt.log_interval == 0:
            l = sum(larr)/len(larr)
            lseg, lemb = sum(lseg_arr) / len(lseg_arr), sum(lemb_arr) / len(lemb_arr)
            pcont = "Step %d, train loss: %.3f, seg loss: %.3f, emb loss: %.3f" % (ckpt['step'], l, lseg, lemb)
            logging.info(pcont)
            open(opt.log_path, 'a+').write(pcont+"\n")
            ckpt['optimizer'] = optimizer.state_dict()
            ckpt['encoder'] = encoder.state_dict()
            if opt.amp == 1:
                ckpt['amp'] = amp.state_dict()
            torch.save(ckpt, open(opt.ckpt_path, 'wb'))
            larr, lseg_arr, lemb_arr = [], [], []
    return

def evaluate(encoder, loader, opt):
    encoder.eval()
    larr, pred_arr, label_arr = [], [], []
    lseg_arr, lemb_arr = [], []
    for i_batch, sample in enumerate(loader):
        feats, labels, label_batch, prob_sizes, label_sizes = sample['feature'], sample['label'], sample['label_batch'], sample['prob_size'], sample['label_size']
        with torch.no_grad():
            W, prob_sizes, l, lseg, lemb = encoder(feats, label_sizes, prob_sizes, label_batch, opt.emb_agwe)

        larr.append(l.item())
        lseg_arr.append(lseg.item())
        lemb_arr.append(lemb.item())

        # Decode
        for j in range(len(W)):
            pred = utils.viterbi(W[j][:prob_sizes[j]])
            pred_arr.append(pred)
            start, end = sum(label_sizes[:j]), sum(label_sizes[:j+1])
            label_arr.append(labels[start: end].tolist())
    acc = lev.compute_acc(pred_arr, label_arr, costs=(1, 1, 1))
    l, lseg, lemb = sum(larr)/len(larr), sum(lseg_arr)/len(lseg_arr), sum(lemb_arr)/len(lemb_arr)
    return l, lseg, lemb, acc

def main():
    parser = options.get_parser()
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logging.info(f"host: {socket.gethostname()}")

    if not os.path.isdir(args.output):
        logging.info("Make dir %s" % (args.output))
        os.makedirs(args.output)

    torch.manual_seed(0)
    np.random.seed(0)

    best_acc_path = os.path.join(args.output, "best-ter.pth") # "./output/conv_lstm.pth"
    log_path = os.path.join(args.output, "log")
    ckpt_path = os.path.join(args.output, "ckpt.pth")

    with open(os.path.join(args.output, 'train_conf.yaml'), 'w') as fo:
        yaml.dump({**vars(args), **{'best_dev_path': best_acc_path}}, fo)
    loader_seeds = list(range(100))

    logging.info(f'max input length: {args.max_ilen[-1]}, max output length: {args.max_olen[-1]}')
    batch_milestones = list(zip(args.max_ilen, args.max_olen, [args.batch_reduce_ratio**i for i in range(1, len(args.max_ilen)+1)]))
    device, cpu = torch.device('cuda'), torch.device('cpu')
    word2id, id2word = utils.get_word_map(args.wordlist)
    assert len(word2id) == len(id2word), "duplicate words"

    train_data = dataset.Speech(args.train_ark, args.train_scp, args.train_text, args.train_len, args.data_sample_rate, args.lstm_sample_rate, add_delta=True, transform=dataset.ToTensor(device), max_segment_size=args.segment_size, add_spec_aug=(args.add_spec_aug==1), min_flen_aug=args.min_flen_aug, spec_F=args.spec_F, spec_T=args.spec_T)
    dev_data = dataset.Speech(args.dev_ark, args.dev_scp, args.dev_text, args.dev_len, args.data_sample_rate, args.lstm_sample_rate, add_delta=True, transform=dataset.ToTensor(device), max_segment_size=args.segment_size)

    if args.batch_sampler == 'utt':
        from sampler import BucketBatchSampler
        train_loader = tud.DataLoader(train_data, batch_sampler=BucketBatchSampler(shuffle=args.shuffle, batch_size=args.batch_size, files=train_data.ark_fns, lengths=train_data.lengths, milestones=batch_milestones, cycle=True, seeds=list(loader_seeds)), collate_fn=dataset.collate_fn)
        dev_loader = tud.DataLoader(dev_data, batch_sampler=BucketBatchSampler(shuffle=False, batch_size=args.dev_batch_size, files=dev_data.ark_fns, lengths=dev_data.lengths, milestones=batch_milestones, cycle=False), collate_fn=dataset.collate_fn)
    else:
        raise NotImplementedError

    logging.info('Number of GPUs %d' % (torch.cuda.device_count()))
    encoder = SegModel(input_size=train_data.feat_dim, hidden_size=args.n_hidden, n_layers=args.n_layers, bid=True, num_label=len(word2id), segment_size=args.segment_size, segment_ratio=args.segment_ratio, sample_rate=args.lstm_sample_rate, dropout=args.dropout, pooling_type=args.pooling, lambda_emb=args.lambda_emb, penalize_emb=args.penalize_emb, word_bias=(args.word_bias==1), num_word_samples=args.seg_word_samples).to(device)
    if args.optim == 'Adam':
        optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum, nesterov=(args.nesterov==1), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('Option for optimizer: Adam|SGD')

    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_gamma, patience=1, min_lr=1.0e-8, verbose=True) # threshold=0.001, 
    else:
        raise NotImplementedError('Option for scheduler: StepLR|ReduceLROnPlateau')
    logging.info(f"{encoder}, {optimizer}, {scheduler}")


    if args.amp == 1:
        logging.info(f"AMP training, opt level: O1")
        encoder, optimizer = amp.initialize(encoder, optimizer, opt_level="O1")

    ckpt = {'epoch': 0, 'step': 0, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_dev_acc': -float('inf'), 'best_dev_loss': float('inf'), 'sampler': train_loader.batch_sampler.state_dict(), 'encoder': encoder.state_dict()}

    if args.amp == 1:
        ckpt['amp'] = amp.state_dict()

    if args.load_awe is not None:
        logging.info('Load awe model %s' % (args.load_awe))
        awe_std = torch.load(args.load_awe)
        encoder.lstm.load_state_dict(awe_std['lstm'])
        encoder.pooling.load_state_dict(awe_std['pooling'])
        encoder.fc1[0].load_state_dict(awe_std['fc1'])

    if args.load_emb is not None:
        logging.info('Unit normalizing word embedding matrix')
        logging.info('Load pre-trained embedding %s' % (args.load_emb))
        emb_agwe = torch.FloatTensor(np.load(args.load_emb)).to(device)
        emb_agwe = emb_agwe / (emb_agwe*emb_agwe).sum(dim=-1, keepdim=True).pow(0.5)
        encoder.fc1[-1].weight.data.copy_(emb_agwe)
    else:
        emb_agwe = None

    if os.path.isfile(ckpt_path):
        logging.info('Load checkpoint %s' % (ckpt_path))
        try:
            ckpt = torch.load(ckpt_path)
            for key, value in ckpt['encoder'].items():
                if key.split('.')[0] != 'seg_loss':
                    encoder.state_dict()[key].copy_(value)
            logging.info(f"Loaded encoder parameters")
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            # load data loader
            train_loader.batch_sampler.load_state_dict(ckpt['sampler'])
            if args.amp == 1:
                amp.load_state_dict(ckpt['amp'])
        except Exception as err:
            logging.info(f"Error loading {ckpt_path}: {err}")

    train_opt = argparse.Namespace(max_grad=args.max_grad, emb_agwe=emb_agwe, accum_batch_size=args.accum_batch_size, log_interval=args.log_interval, log_path=log_path, ckpt_path=ckpt_path, amp=args.amp)
    dev_opt = argparse.Namespace(emb_agwe=emb_agwe)

    for epoch in range(args.epoch):
        if epoch < ckpt['epoch']:
            continue
        logging.info('Epoch %d, lr: %.5f' % (epoch, optimizer.param_groups[0]['lr']))
        train(encoder, optimizer, train_loader, ckpt, train_opt)
        dl, dl_seg, dl_emb, dacc = evaluate(encoder, dev_loader, dev_opt)
        if args.scheduler == 'StepLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(dacc)
        else:
            raise NotImplementedError('Option for scheduler: StepLR|ReduceLROnPlateau')
        pcont = 'Epoch %d, dev loss: %.3f, seg loss: %.3f, emb loss: %.3f, acc (OOV): %.3f' % (epoch, dl, dl_seg, dl_emb, dacc)
        logging.info(pcont)
        open(log_path, 'a+').write(pcont+"\n")
        # if dacc > ckpt['best_dev_acc']:
        #     ckpt['best_dev_acc'] = dacc
        #     torch.save(encoder.state_dict(), open(best_acc_path, 'wb'))
        ckpt['epoch'] = ckpt['epoch'] + 1
        ckpt['encoder'] = encoder.state_dict()
        ckpt['optimizer'] = optimizer.state_dict()
        ckpt['scheduler'] = scheduler.state_dict()
        if args.amp == 1:
            ckpt['amp'] = amp.state_dict()
        if dacc > ckpt['best_dev_acc']:
            ckpt['best_dev_acc'] = dacc
            torch.save(encoder.state_dict(), open(best_acc_path, 'wb'))
            torch.save(ckpt, open(ckpt_path+'.best', 'wb'))
        torch.save(ckpt, open(ckpt_path, 'wb'))
    return

if __name__ == '__main__':
    main()
