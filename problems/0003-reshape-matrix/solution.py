import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	if not a or not a[0]:
		return []
	if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return []
    # 1. 拍平成一维列表: [[1, 2], [3, 4]] -> [1, 2, 3, 4]
    flat = [x for row in a for x in row]
    
    rows, cols = new_shape
    # 2. 按照每行 cols 个元素进行切片组装
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]