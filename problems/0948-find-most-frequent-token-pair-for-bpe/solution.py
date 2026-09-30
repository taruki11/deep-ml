def most_frequent_pair(sequences):
    count={}
    for seq in sequences:
        for i in range(len(seq)-1):
            pair=(seq[i],seq[i+1])
            count[pair]=count.get(pair,0)+1
    if not count:
        return None
    return max(count,key=count.get)
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    pass