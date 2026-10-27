def iou(box_a, box_b):
    # box 格式: [x1, y1, x2, y2]，保证 x1 < x2, y1 < y2
    # 1. 计算交集矩形的左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    # 2. 计算交集宽度与高度（小于 0 说明不相交，截断为 0.0）
    inter_w = max(0.0, float(x2 - x1))
    inter_h = max(0.0, float(y2 - y1))
    inter_area = inter_w * inter_h
    
    if inter_area == 0.0:
        return 0.0
        
    # 3. 计算各自的面积与并集面积
    area_a = float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1])
    area_b = float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0.0:
        return 0.0
        
    return round(inter_area / union_area, 4)

def mean_average_precision(preds, gts, iou_thresh=0.5):
    # 1. 提取所有在真实标注 (gts) 中出现过的类别
    gt_classes = sorted(list(set(c for c, _ in gts)))
    if not gt_classes:
        return 0.0
        
    aps = []
    
    # 2. 对每个类别独立计算 AP
    for cls in gt_classes:
        # 该类别的真实框列表
        cls_gts = [box for c, box in gts if c == cls]
        num_gts = len(cls_gts)
        
        # 该类别的预测框列表 (class_id, score, box)
        cls_preds = [p for p in preds if p[0] == cls]
        
        # 若该类别模型没有任何预测，AP 记为 0
        if not cls_preds:
            aps.append(0.0)
            continue
            
        # 按照预测置信度由大到小降序排列
        cls_preds.sort(key=lambda p: p[1], reverse=True)
        
        # 标记真实框是否已经被某个预测框匹配过（每个 GT 框至多匹配一次）
        gt_used = [False] * num_gts
        
        tp = []
        fp = []
        
        # 3. 逐个预测框匹配未使用的 GT 框
        for p in cls_preds:
            p_box = p[2]
            best_iou = -1.0
            best_gt_idx = -1
            
            # 在所有尚未被匹配的真实框中，寻找重合度 (IoU) 最大的那一个
            for gt_idx, gt_box in enumerate(cls_gts):
                if not gt_used[gt_idx]:
                    cur_iou = iou(p_box, gt_box)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_gt_idx = gt_idx
                        
            # 如果最佳 IoU 达到阈值，判定为 True Positive 并标记该 GT 已消耗
            if best_gt_idx != -1 and best_iou >= iou_thresh:
                tp.append(1)
                fp.append(0)
                gt_used[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
                
        # 4. 计算逐步累加的 Precision 和 Recall 序列
        num_preds = len(cls_preds)
        precisions = []
        recalls = []
        
        cum_tp = 0
        cum_fp = 0
        for i in range(num_preds):
            cum_tp += tp[i]
            cum_fp += fp[i]
            # Precision = 当前识别出的真实正样本数 / 当前一共预测的正样本数
            precisions.append(cum_tp / (i + 1))
            # Recall = 当前识别出的真实正样本数 / 图像中总共存在的真实样本数
            recalls.append(cum_tp / num_gts)
            
        # 5. 从右向左使 Precision 单调非递增: p[i] = max(p[i:])
        for i in range(num_preds - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
            
        # 6. 全点积分法计算当前类别的 AP: sum_i (r_i - r_{i-1}) * p_i, r_{-1} = 0
        ap = 0.0
        for i in range(num_preds):
            prev_r = recalls[i - 1] if i > 0 else 0.0
            delta_r = recalls[i] - prev_r
            ap += delta_r * precisions[i]
            
        aps.append(ap)
        
    # 7. 计算所有类别的平均 AP (mAP)，保留 4 位小数
    mAP = sum(aps) / len(gt_classes)
    return round(mAP, 4)