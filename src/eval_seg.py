import os
import yaml
import configargparse
import torch
import logging
import utils
import kaldi_io
import numpy as np
import torch.utils.data as tud
from torch import nn
import delta_ark as dataset
from sampler import BucketBatchSampler
from seg_model import SegModel

def get_trans(encoder, loader, output_fn, id2word):
    encoder.eval()
    pred_arr, utt_ids = [], []
    for i_batch, sample in enumerate(loader):
        feats, labels, label_batch, prob_sizes, label_sizes = sample['feature'], sample['label'], sample['label_batch'], sample['prob_size'], sample['label_size']
        with torch.no_grad():
            W, prob_sizes, l, lseg, lemb = encoder(feats, label_sizes, prob_sizes, label_batch, emb_agwe=None)
        # Decode
        for j in range(len(W)):
            pred = utils.viterbi(W[j][:prob_sizes[j]])
            pred = ' '.join([id2word[int_id] for int_id in pred])
            pred_arr.append(pred)
            utt_ids.append(sample['utt_id'][j])
    logging.info(f'Writing predictions to {output_fn}')
    with open(output_fn, 'w') as fo:
        for ix in range(len(pred_arr)):
            fo.write(utt_ids[ix] + ' ' + pred_arr[ix] + '\n')
    return

def main():
    parser = configargparse.ArgumentParser(
        description="Eval Seg Model",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_train', is_config_file=False, help='train config file path')
    parser.add_argument('--config_eval', is_config_file=False, help='eval config file path')
    parser.add_argument("--eval_ark", type=str, help="ark dir")
    parser.add_argument("--eval_scp", type=str, help="scp file")
    parser.add_argument("--eval_text", type=str, help="text file")
    parser.add_argument("--eval_len", type=str, help="length file")
    parser.add_argument("--eval_out", type=str, help="output trans file")
    parser.add_argument("--eval_type", type=str, default='word', help="eval type")
    args, _ = parser.parse_known_args()
    for config_file in [args.config_train, args.config_eval]:
        rem_args = yaml.safe_load(open(config_file, 'r'))
        parser.set_defaults(**rem_args)
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    torch.manual_seed(0)
    np.random.seed(0)
    logging.info(f'max input length: {args.max_ilen[-1]}, max output length: {args.max_olen[-1]}')
    batch_milestones = list(zip(args.max_ilen, args.max_olen, [args.batch_reduce_ratio**i for i in range(1, len(args.max_ilen)+1)]))
    device, cpu = torch.device('cuda'), torch.device('cpu')
    word2id, id2word = utils.get_word_map(args.wordlist)

    data = dataset.Speech(args.eval_ark, args.eval_scp, args.eval_text, args.eval_len, args.data_sample_rate, args.lstm_sample_rate, add_delta=True, transform=dataset.ToTensor(device), max_segment_size=args.segment_size, train=False)
    loader = tud.DataLoader(data, batch_sampler=BucketBatchSampler(shuffle=False, batch_size=args.dev_batch_size, files=data.ark_fns, lengths=data.lengths, milestones=batch_milestones, cycle=False), collate_fn=dataset.collate_fn)
    print('Number of GPUs', torch.cuda.device_count())
    encoder = SegModel(input_size=data.feat_dim, hidden_size=args.n_hidden, n_layers=args.n_layers, bid=True, num_label=len(word2id), segment_size=args.segment_size, segment_ratio=args.segment_ratio, sample_rate=args.lstm_sample_rate, dropout=args.dropout, pooling_type=args.pooling, lambda_emb=args.lambda_emb, penalize_emb=args.penalize_emb, word_bias=(args.word_bias==1), num_word_samples=-1).to(device)
    logging.info('Loading %s' % (args.best_dev_path))
    model_state_dict = torch.load(args.best_dev_path)
    for pname, pval in model_state_dict.items():
        if pname.split('.')[0] == 'seg_loss':
            continue
        encoder.state_dict()[pname].copy_(pval)
    logging.info(f"{encoder}")
    get_trans(encoder, loader, args.eval_out, id2word)
    return

if __name__ == '__main__':
    main()
