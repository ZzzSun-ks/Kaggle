import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

import config
from dataset import get_dataloader
from model import TimeSeriesModel


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for inputs, targets in tqdm(dataloader, desc='Train'):
        inputs = inputs.to(device)  # [batch_size, seq_len]
        targets = targets.to(device)  # [batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.shape[0]
        total_samples += targets.shape[0]
    return total_loss / total_samples


def validate(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs = inputs.to(device)  # [batch_size, seq_len]
        targets = targets.to(device)  # [batch_size]

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        total_loss += loss.item() * inputs.shape[0]
        total_samples += targets.shape[0]
    return total_loss / total_samples


def train():
    # 1.设备
    device = torch.device('cpu')

    # 2.数据
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader()
    sample_batch = next(iter(train_dataloader))
    input, target = sample_batch

    # 3.模型
    model = TimeSeriesModel(input_size=input.shape[2],
                            hidden_size=config.HIDDEN_SIZE,
                            num_layers=1,
                            dropout=0.2).to(device)

    # 4.损失函数
    loss_function = nn.MSELoss()

    # 5.优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6.学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=2,
                                                           threshold=1e-4,
                                                           min_lr=1e-7
                                                           )

    # 7.早停参数
    patience_counter = 0
    early_stop_patience = 5

    # 8.TensorBoard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'============= EPOCH {epoch} =============')
        train_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer, device)
        val_loss = validate(model, test_dataloader, loss_function, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'train_loss: {train_loss}, val_loss: {val_loss}, lr: {current_lr}')

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)

        # 保存模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print('保存模型')
        else:
            patience_counter += 1
            print(f'早停计数: {patience_counter}/{early_stop_patience}')

        if patience_counter >= early_stop_patience:
            print(f'早停触发！验证损失连续{early_stop_patience}个epoch未改善')
            break

    writer.close()


if __name__ == "__main__":
    train()
