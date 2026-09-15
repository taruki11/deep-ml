import re

def tokenize_text(text: str) -> list[str]:
    # 分隔符规则：双破折号 | 标点符号集 | 1个或多个连续空白
    pattern = r'(--|[,.:;?_!"\(\)\']|\s+)'

    # 1. 切分（括号保证分隔符本身也被切出）
    pieces = re.split(pattern, text)

    # 2. 清洗：strip 并剔除所有变成空串的切片
    tokens = []
    for piece in pieces:
        cleaned = piece.strip()
        if cleaned:  # 等价于 if cleaned != ""
            tokens.append(cleaned)

    return tokens