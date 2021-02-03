def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """ 
    iterative_levenshtein(s, t) -> ldist
    ldist is the Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein 
    distance between the first i characters of s and the 
    first j characters of t

    s: source, t: target
    costs: a tuple or a list with three integers (d, i, s)
           where d defines the costs for a deletion
                 i defines the costs for an insertion and
                 s defines the costs for a substitution
    return: 
    H, S, D, I: correct chars, number of substitutions, number of deletions, number of insertions
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs
    
    dist = [[0 for x in range(cols)] for x in range(rows)]
    H, D, S, I = 0, 0, 0, 0
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * inserts
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * deletes
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + inserts,
                                 dist[row][col-1] + deletes,
                                 dist[row-1][col-1] + cost) # substitution
    row, col = rows-1, cols-1
    t_aligns, s_aligns = [], []
    while row != 0 or col != 0:
        if row == 0:
            D += col
            s_aligns.extend(['*']*col)
            t_aligns.extend(t[:col][::-1])
            col = 0
        elif col == 0:
            I += row
            t_aligns.extend(['*']*row)
            s_aligns.extend(s[:row][::-1])
            row = 0
        elif dist[row][col] == dist[row-1][col] + inserts:
            t_aligns.append('*')
            I += 1
            row = row-1
            s_aligns.append(s[row])
        elif dist[row][col] == dist[row][col-1] + deletes:
            s_aligns.append('*')
            D += 1
            col = col-1
            t_aligns.append(t[col])
        elif dist[row][col] == dist[row-1][col-1] + substitutes:
            S += 1
            row, col = row-1, col-1
            s_aligns.append(s[row])
            t_aligns.append(t[col])
        else:
            H += 1
            row, col = row-1, col-1
            s_aligns.append(s[row])
            t_aligns.append(t[col])

    s_aligns, t_aligns = s_aligns[::-1], t_aligns[::-1]
    return H, D, S, I, s_aligns, t_aligns

def compute_acc(preds, labels, costs=(1, 1, 1)):
    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')
    Ns, Ds, Ss, Is = 0, 0, 0, 0
    for i, _ in enumerate(preds):
        H, D, S, I, _, _ = iterative_levenshtein(preds[i], labels[i], costs)
        Ns += len(labels[i])
        Ds += D
        Ss += S
        Is += I
    try:
        acc = 100*(Ns-Ds-Ss-Is)/Ns
    except ZeroDivisionError as err:
        raise ZeroDivisionError('Empty labels')
    return acc
