def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if not matrix or not matrix[0]:
		return []
	row=len(matrix)
	col=len(matrix[0])
	if mode=='row':
		return [sum(i)/col for i in matrix]
	else:
		return [sum(j)/row for j in zip(*matrix)]
