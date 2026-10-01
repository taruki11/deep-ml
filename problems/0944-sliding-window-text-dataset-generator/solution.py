def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    pair=[]
    n=len(token_ids)
    for i in range(0,n-max_length,stride):
        input_chunk=token_ids[i:i+max_length]
        output_chunk=token_ids[i+1:i+1+max_length]
        pair.append((input_chunk,output_chunk))
    return pair
    
    pass