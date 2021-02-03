import torch
import logging
import torch.nn as nn
import torch.nn.functional as functional
from jit_segloss import SegLossModule
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

INT_MAX = 2147483647

class SegModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bid, num_label, segment_size, segment_ratio, sample_rate, dropout, pooling_type='concat', lambda_emb=0, penalize_emb='batch', word_bias=False, num_word_samples=5000):
        super(SegModel, self).__init__()
        self.n_layers, self.nhid, self.nin, self.num_direction = n_layers, hidden_size, input_size, int(bid)+1
        self.num_label, self.segment_size, self.sample_rate = num_label, segment_size, sample_rate
        self.segment_ratio = segment_ratio
        logging.info('Segment size: %d, Model sample rate: %d' % (segment_size, sample_rate))
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=bid, dropout=dropout, batch_first=True)
        self.pooling = nn.Sequential(
            nn.Conv1d(hidden_size*(int(bid)+1), hidden_size, kernel_size=sample_rate+1, stride=1, padding=sample_rate//2),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=sample_rate),
            nn.Dropout(dropout))
        self.pooling_type = pooling_type
        if self.pooling_type == 'concat':
            logging.info(f'bias term in embedding layer: {word_bias}')
            fdim = hidden_size*2
            self.fc1 = nn.Sequential(
                nn.Linear(fdim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, self.num_label, bias=word_bias))
        else:
            raise NotImplementedError
        self.lambda_emb, self.penalize_emb, self.num_word_samples = lambda_emb, penalize_emb, num_word_samples
        logging.info(f'lambda on emb loss: {lambda_emb}, penalize: {penalize_emb}, word samples: {num_word_samples}')
        self.seg_loss = SegLossModule(chunk_size=128, verbose=False)

    def forward(self, Xs, label_sizes, frame_sizes, label_batch, emb_agwe=None):
        """
        Xs: (B, L, F), label_sizes/frame_sizes: (B)
        """
        S = int(min(self.segment_size, self.segment_ratio * (frame_sizes / (label_sizes * self.sample_rate)).max().item())) if self.training else self.segment_size

        h0 = self.init_hidden(len(Xs))
        self.lstm.flatten_parameters()

        Xs = pack_padded_sequence(Xs, frame_sizes, batch_first=True, enforce_sorted=False)
        Fs, _ = self.lstm(Xs, h0) # [B, L, F]
        Fs, _ = pad_packed_sequence(Fs, batch_first=True)
        Fs = Fs.transpose(1, 2)
        Fs = self.pooling(Fs).transpose(1, 2)

        L = Fs.size(1)
        frame_sizes = frame_sizes / self.sample_rate

        Fs_pad = torch.cat((Fs, Fs.new_zeros(Fs.size(0), S, Fs.size(-1))), dim=1) # [B, L+S, F]

        # ix: start ids, iy: end ids
        ix = torch.arange(L).unsqueeze(dim=-1).expand(-1, S).contiguous() # (L, S)
        iy = ix + torch.arange(S).unsqueeze(dim=0) # (L, S)
        ix, iy = ix.view(-1), iy.view(-1)

        if self.pooling_type == 'concat':
            fx, fy = Fs_pad[:, ix, :], Fs_pad[:, iy, :]
            fxy = torch.cat((fx, fy), dim=-1)
            logprob = self.fc1[:-1](fxy)
        else:
            raise NotImplementedError

        if self.training and self.num_word_samples > 0:
            label_batch_ = -torch.ones_like(label_batch)
            origin_label_batch = label_batch
            label_batch = label_batch.tolist()
            label_samples = torch.randperm(self.num_label).tolist()[:self.num_word_samples] + [label_batch[i_batch][i_label] for i_batch in range(len(label_batch)) for i_label in range(label_sizes[i_batch])]
            old2new = {}
            label_id = 0
            label_samples = list(Counter(label_samples))
            old2new = dict([(old_label, i_label) for i_label, old_label in enumerate(label_samples)])
            bias_sample = self.fc1[-1].bias[label_samples] if self.fc1[-1].bias is not None else self.fc1[-1].weight.new_zeros(1)
            fc_embed = self.fc1[-1].weight[label_samples, :]
            logprob = torch.matmul(logprob, fc_embed.transpose(0, 1)) + bias_sample
            logprob = logprob.view(logprob.size(0), L, S, len(label_samples))
            for i_batch in range(len(label_batch)):
                for i_label in range(len(label_batch[0])):
                    if i_label < label_sizes[i_batch]:
                        label_batch_[i_batch, i_label] = old2new[label_batch[i_batch][i_label]]
                    else:
                        label_batch_[i_batch, i_label] = label_batch[i_batch][i_label]
            label_batch = label_batch_
        else:
            origin_label_batch = label_batch
            logprob = self.fc1[-1](logprob)
            logprob = logprob.view(logprob.size(0), L, S, self.num_label)

        W, prob_sizes = logprob, frame_sizes
        assert W.size(0) * W.size(1) * W.size(2) * W.size(3) < INT_MAX

        label_batch, label_sizes, prob_sizes = label_batch.type(torch.int).to(W.device), label_sizes.type(torch.int).to(W.device), prob_sizes.type(torch.int).to(W.device)
        prob_sizes = torch.min(torch.max(prob_sizes, label_sizes), S*label_sizes)
        prob_sizes = prob_sizes.clamp(max=W.size(1))
        l = self.seg_loss(W, label_batch, label_sizes, prob_sizes, reduction='SUM')
        if emb_agwe is not None:
            loss_emb = ((self.fc1[-1].weight - emb_agwe) ** 2).mean(dim=-1)
            if self.penalize_emb == 'all':
                loss_emb = loss_emb.sum()
            elif self.penalize_emb == 'batch':
                mask = torch.zeros_like(loss_emb)
                labels = torch.cat([origin_label_batch[b, :label_sizes[b]] for b in range(origin_label_batch.size(0))]).type(torch.long)
                labels_non_unk = labels[labels != self.num_label-1]
                mask[labels_non_unk] = 1
                loss_emb = loss_emb[mask == 1].sum()
            else:
                raise NotImplementedError
        else:
            loss_emb = W.new_zeros(1)
        return logprob, frame_sizes, (1-self.lambda_emb) * l + self.lambda_emb * loss_emb, l, loss_emb

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers*self.num_direction, bsz, self.nhid),
                weight.new_zeros(self.n_layers*self.num_direction, bsz, self.nhid))
