import os
import gzip
import torch
import tempfile
import kaldi_io
import logging
import numpy as np
import torch.utils.data as tud
from spec_augment import specaug

class Speech(tud.Dataset):
    def __init__(self, ark_dir, scp_fn, label_fn, len_fn, data_sr=2, model_sr=1, add_delta=True, transform=None, max_segment_size=32, train=True, add_spec_aug=False, min_flen_aug=10, spec_F=20, spec_T=40):
        self.ark_dir = ark_dir
        self.data_sr, self.model_sr = data_sr, model_sr
        self.max_segment_size = max_segment_size
        logging.info('Data sample rate: %d, model sample rate: %d' % (data_sr, model_sr))
        self.add_delta = add_delta
        self.add_spec_aug = add_spec_aug
        self.min_flen_aug = min_flen_aug
        self.spec_F, self.spec_T = spec_F, spec_T
        logging.info('Spec aug: {}, T: {}, F: {}'.format(add_spec_aug, spec_T, spec_F))
        self.transform = transform
        self.open_ark = None
        self.train = train
        self._parse(scp_fn, label_fn, len_fn)

    def __getitem__(self, idx):
        utt_id, ark_fn, label = self.utt_ids[idx], self.ark_fns[idx], self.labels[idx]
        ark_fn = os.path.join(self.ark_dir, ark_fn)
        if ark_fn != self.open_ark:
            self.open_ark = ark_fn
            logging.info(f"Open ark: {self.open_ark}")
            utt_to_feat = {key: mat for key, mat in kaldi_io.read_mat_ark(self.open_ark)}
            for utt_id, feat in utt_to_feat.items():
                with torch.no_grad():
                    if self.train and self.add_spec_aug and len(feat) > self.min_flen_aug:
                        feat = specaug(torch.from_numpy(feat), F=self.spec_F, T=self.spec_T).cpu().numpy() # .cuda()
                utt_to_feat[utt_id] = feat
            tmp_ark = tempfile.mkdtemp() + '/tmp.ark'
            ark_output = 'ark:| copy-feats --compress=true ark:- ark:' + tmp_ark
            with kaldi_io.open_or_fd(ark_output, 'wb') as fo:
                for key, mat in utt_to_feat.items():
                    kaldi_io.write_mat(fo, mat, key=key)
            if self.add_delta:
                fd = 'add-deltas ark:' + tmp_ark + ' ark:-|'
            else:
                fd = tmp_ark
            self.utt_to_feat = {key: mat for key, mat in kaldi_io.read_mat_ark(fd)}

        utt_id = self.utt_ids[idx]
        feat = self.utt_to_feat[utt_id]
        plen = len(feat) % self.data_sr
        if plen != 0:
            feat = np.concatenate((feat, np.zeros((self.data_sr-plen, feat.shape[1]), dtype=np.float32)), axis=0)
        feat = feat.reshape(-1, self.data_sr, feat.shape[-1]).reshape(-1, self.data_sr*feat.shape[-1])
        if len(feat) < len(label):
            logging.info('%s in %s: Feature size < label size' % (utt_id, ark_fn))
            # zero padding
            feat = np.concatenate((feat, np.zeros((len(label)-len(feat), feat.shape[1]), dtype=np.float32)), axis=0)
        max_feat_size = len(label) * self.max_segment_size * self.model_sr
        if len(feat) > max_feat_size and self.train:
            logging.info("Truncating from %d to %d" % (len(feat), max_feat_size))
            feat = feat[:max_feat_size]
        sample = {'utt_id': utt_id, 'feature': feat, 'label': label, 'prob_size': [len(feat)], 'label_size': [len(label)]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.utt_ids)

    def _parse(self, scp_fn, label_fn, len_fn):
        scp_lns = open(scp_fn, 'r').readlines()
        label_lns = gzip.open(label_fn, 'rb').readlines()
        len_lns = open(len_fn, 'r').readlines()
        utt_to_label = {}
        for ln in label_lns:
            ln = ln.decode('utf-8').split()
            utt_id, label = ln[0], list(map(lambda x: int(x), ln[1:]))
            utt_to_label[utt_id] = label
        utt_to_length = {}
        for ln in len_lns:
            utt_id, input_length = ln.strip().split()
            output_length = len(utt_to_label[utt_id])
            input_length = int(input_length)
            utt_to_length[utt_id] = (input_length, output_length)
        self.utt_ids, self.ark_fns, self.labels, self.lengths = [], [], [], []
        for ln in scp_lns:
            utt_id, ark_fn = ln.split()
            self.utt_ids.append(utt_id)
            ark_fn = ark_fn.split(':')[0]
            self.ark_fns.append(ark_fn)
            self.labels.append(utt_to_label[utt_id])
            self.lengths.append(utt_to_length[utt_id])
        self.feat_dim = next(kaldi_io.read_mat_ark(os.path.join(self.ark_dir, ark_fn)))[1].shape[1]
        if self.add_delta:
            self.feat_dim = 3 * self.feat_dim
        self.feat_dim = self.data_sr * self.feat_dim
        logging.info('Number of utts %d, feat dimension %d' % (len(self.utt_ids), self.feat_dim))
        return

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        feat, label, prob_size, label_size = sample['feature'], sample['label'], sample['prob_size'], sample['label_size']
        feat, label = torch.from_numpy(feat).float().to(self.device), torch.LongTensor(label)
        prob_size, label_size = torch.IntTensor(prob_size), torch.IntTensor(label_size)
        sample = {'feature': feat, 'label': label, 'prob_size': prob_size, 'label_size': label_size, 'utt_id': sample['utt_id']}
        return sample

def collate_fn(data):
    bsz, Nmax, Lmax, imsz = len(data), max([x['feature'].size(0) for x in data]), max([x['label'].size(0) for x in data]), list(data[0]['feature'].size()[1:])
    frames = data[0]['feature'].new_zeros(*([bsz, Nmax] + imsz))
    labels, prob_sizes, label_sizes, utt_ids = [], [], [], []
    label_batch = -torch.ones(bsz, Lmax).type(torch.IntTensor)
    for i in range(bsz):
        frames[i, :data[i]['feature'].size(0), :] = data[i]['feature']
        label_batch[i, :data[i]['label_size']] = data[i]['label']
        labels.append(data[i]['label'])
        prob_sizes.append(data[i]['prob_size'])
        label_sizes.append(data[i]['label_size'])
        utt_ids.append(data[i]['utt_id'])
    labels, prob_sizes, label_sizes = torch.cat(labels), torch.cat(prob_sizes), torch.cat(label_sizes)
    sample = {'feature': frames, 'label': labels, 'prob_size': prob_sizes, 'label_size': label_sizes, 'label_batch': label_batch, 'utt_id': utt_ids}
    return sample
