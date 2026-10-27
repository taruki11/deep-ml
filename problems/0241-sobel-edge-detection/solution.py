import math

def sobel_edge_detection(image):
    # 1. 边界与类型校验：不合法时直接返回 -1
    if image is None:
        return -1
    
    # 支持传入 numpy 数组，统一转成 list
    if hasattr(image, 'tolist'):
        try:
            image = image.tolist()
        except Exception:
            return -1
            
    if not isinstance(image, (list, tuple)):
        return -1
    
    H = len(image)
    if H < 3:
        return -1  # 图像高度不足 3 行
    
    W = None
    for row in image:
        if hasattr(row, 'tolist'):
            try:
                row = row.tolist()
            except Exception:
                return -1
        if not isinstance(row, (list, tuple)):
            return -1
        if W is None:
            W = len(row)
            if W < 3:
                return -1  # 图像宽度不足 3 列
        elif len(row) != W:
            return -1  # 锯齿形状（各行长度不一致）
            
        for val in row:
            # 排除布尔值（Python 中 bool 是 int 的子类）和非数值，以及越界数值
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return -1
            if val < 0 or val > 255:
                return -1

    # 2. 定义标准 Sobel 3x3 卷积核
    Gx_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    Gy_kernel = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    out_h = H - 2
    out_w = W - 2
    
    # 3. 滑动窗口卷积，计算梯度幅值
    magnitudes = []
    max_mag = 0.0
    
    for i in range(out_h):
        mag_row = []
        for j in range(out_w):
            gx = 0.0
            gy = 0.0
            for ki in range(3):
                for kj in range(3):
                    pixel = image[i + ki][j + kj]
                    gx += pixel * Gx_kernel[ki][kj]
                    gy += pixel * Gy_kernel[ki][kj]
            
            mag = math.sqrt(gx * gx + gy * gy)
            mag_row.append(mag)
            if mag > max_mag:
                max_mag = mag
        magnitudes.append(mag_row)
        
    # 4. 基于最大幅值归一化至 [0, 255] 并四舍五入转为整型
    result = []
    for i in range(out_h):
        res_row = []
        for j in range(out_w):
            if max_mag > 0:
                val = (magnitudes[i][j] / max_mag) * 255.0
                res_row.append(int(round(val)))
            else:
                res_row.append(0)
        result.append(res_row)
        
    return result