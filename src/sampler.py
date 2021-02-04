import torch
import logging
import numpy as np
import torch.utils.data as tud
from collections import defaultdict

class BucketBatchSampler(tud.Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    Example:
        >>> list(BucketBatchSampler(shuffle=True, batch_size=2, files=['f0']*3+['f1']*5))
        [[7, 5], [3, 4], [6], [1, 2], [0]]
    """
    def __init__(self, shuffle, batch_size, files, lengths, milestones, cycle=True, seeds=list(range(100))):
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be a boolean value,"
                             "but got shuffle={}"
                             .format(shuffle))
        if not (isinstance(batch_size, int)):
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(files, list):
            raise ValueError("files should be a list type, but got "
                             "files={}".format(files))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files = files
        fidx = defaultdict(list)
        self.idx2len = {}
        for ix, f in enumerate(files):
            fidx[f].append([ix, lengths[ix][0], lengths[ix][1]])
            self.idx2len[ix] = (lengths[ix][0], lengths[ix][1])
        self.fidx = list(fidx.values())
        self.cycle = cycle
        self._ilen_to_maxbatch, self._olen_to_maxbatch = {}, {}
        prev_ilen, prev_olen, prev_ratio = 1, 1, 1
        logging.info("Loader (input size, output size, reduce ratio): {}".format(milestones))
        for ilen, olen, ratio in milestones:
            for j in range(prev_ilen, ilen):
                self._ilen_to_maxbatch[j] = max(int(batch_size * prev_ratio), 1)
            for j in range(prev_olen, olen):
                self._olen_to_maxbatch[j] = max(int(batch_size * prev_ratio), 1)
            prev_ilen, prev_olen, prev_ratio = ilen, olen, ratio
        logging.info("Initialize seeds and index array")
        seed = seeds.pop()
        batch_indexes = self.make_batches(self.fidx, seed, cycle, shuffle)
        self.state_dict_ = {'seeds': seeds, 'batch_indexes': batch_indexes}

    def __iter__(self):
        if len(self.state_dict_['batch_indexes']) == 0:
            seed = self.state_dict_['seeds'].pop() if len(self.state_dict_['seeds']) > 0 else 222
            logging.info(f"Making new batch from new seed {seed}")
            self.state_dict_['batch_indexes'] = self.make_batches(self.fidx, seed, self.cycle, self.shuffle)
        while True:
            if len(self.state_dict_['batch_indexes']) > 0:
                batch = self.state_dict_['batch_indexes'].pop()
                yield batch
            else:
                break

    def __len__(self):
        logging.info(f"Sampler length {len(self.state_dict_['batch_indexes'])}")
        return len(self.state_dict_['batch_indexes'])

    def make_batches(self, file_indexes, seed, cycle, shuffle):
        perm = []
        batch_indexes = []
        if shuffle:
            np.random.seed(seed)
            xs = np.random.permutation(len(file_indexes))
            for x in xs:
                perm.append(np.random.permutation(file_indexes[x]).tolist())
        else:
            perm = file_indexes
        for index_group in perm:
            batch, ilens, olens = [], [], []
            max_ilen, max_olen, max_batch_size = 1, 1, self._ilen_to_maxbatch[1]
            for index, ilen, olen in index_group:
                batch.append(index)
                ilens.append(ilen)
                olens.append(olen)
                if ilen not in self._ilen_to_maxbatch or olen not in self._olen_to_maxbatch:
                    max_ilen, max_olen = max(list(self._ilen_to_maxbatch.keys())), max(list(self._olen_to_maxbatch.keys()))
                    raise ValueError("In batch loader, Input length %d (max: %d) or output length %d (max: %d) out of milestone-bound" % (ilen, max_ilen, olen, max_olen))
                max_batch_size = min(self._ilen_to_maxbatch[ilen], self._olen_to_maxbatch[olen], max_batch_size)
                if len(batch) >= max_batch_size:
                    next_batch = batch[:len(batch)-max_batch_size]
                    batch_indexes.append(batch[len(batch)-max_batch_size:])
                    batch = next_batch
                    ilens = ilens[:len(batch)-max_batch_size]
                    olens = olens[:len(batch)-max_batch_size]
                    max_ilen = max(ilens) if len(ilens) > 0 else 1
                    max_olens = max(olens) if len(olens) > 0 else 1
                    max_batch_size = self._ilen_to_maxbatch[1]

            if len(batch) > 0:
                if cycle:
                    max_batch_size = min([min(self._ilen_to_maxbatch[self.idx2len[idx][0]], self._olen_to_maxbatch[self.idx2len[idx][1]]) for idx in batch])
                    for i in range(len(batch)//max_batch_size):
                        batch_indexes.append(batch[i*max_batch_size: (i+1)*max_batch_size])
                    if len(batch) % max_batch_size > 0:
                        # possible to have large batches here
                        batch = batch[-(len(batch) % max_batch_size):]
                        res_batch = [g[0] for g in index_group[:max_batch_size - len(batch)]]
                        batch = batch + res_batch
                batch_indexes.append(batch)
        return batch_indexes

    def load_state_dict(self, state_dict):
        self.state_dict_ = state_dict
        logging.info("Loading sampler state dict, %d batches, %d seeds" % (len(self.state_dict_['batch_indexes']), len(self.state_dict_['seeds'])))
        return

    def state_dict(self):
        return self.state_dict_
