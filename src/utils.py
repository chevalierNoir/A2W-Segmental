import torch

def get_word_map(wordlist):
    # word, id
    lns = map(lambda x: x.strip().split(), open(wordlist, 'r', encoding='utf-8').readlines())
    word2id, id2word = {}, {}
    for ln in lns:
        w, id = ln[0], int(ln[1])
        word2id[w] = id
        id2word[id] = w
    return word2id, id2word

def viterbi(W):
    # W: [T, S, V]
    T, S, V = W.size()
    d, delta = W.new_zeros(T+1), [0 for _ in range(T+1)]
    for t in range(1, T+1):
        start = max(0, t-S)
        heads = torch.arange(start, t).type(torch.LongTensor)
        weights = d[heads].view(-1, 1) + W[heads, t-heads-1, :]
        max_col_ids = weights.max(dim=-1)[1]
        row_id = weights[torch.arange(len(heads)).type(torch.LongTensor), max_col_ids].argmax()
        col_id = max_col_ids[row_id]
        d[t] = weights[row_id, col_id]
        delta[t] = (heads[row_id].item(), col_id.item())
    # backtrack
    u, ps = T, []
    while u != 0:
        ps.append(delta[u][1])
        u = delta[u][0]
    ps = ps[::-1]
    return ps
