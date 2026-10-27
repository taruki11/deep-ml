# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    Args:
        model: nn.Module
        X_train: torch.Tensor, shape (N, ...)
        y_train: torch.Tensor, shape (N,)
        X_val: torch.Tensor, shape (M, ...)
        y_val: torch.Tensor, shape (M,)
        epochs: int
        batch_size: int
        lr: float
    
    Returns:
        history: List of dicts, keys: 'epoch', 'train_loss', 'val_loss', 'val_accuracy'
    """
    # 1. 初始化优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    n_samples = len(X_train)
    
    for epoch in range(1, epochs + 1):
        # 2. 训练模式
        model.train()
        
        # 每个 Epoch 开始前对训练数据进行随机打乱
        perm = torch.randperm(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Mini-batch 迭代
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # 标准 5 步训练流程
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / n_batches
        
        # 3. 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_preds = val_outputs.argmax(dim=1)
            val_accuracy = (val_preds == y_val).float().mean().item()
        
        # 4. 记录当前 Epoch 指标 (epoch 从 1 开始计数)
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
    
    return history