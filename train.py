import time
import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    total_data_time = 0.0
    total_train_time = 0.0
    epoch_start_time = time.perf_counter()  # C2.3

    # Data-loading start point
    start_data_time = time.perf_counter()

    pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)

    # use iter for data loading time count
    for inputs, labels in pbar:
        # --- (C2.1) Data-loading time ---

        current_data_time = time.perf_counter() - start_data_time
        total_data_time += current_data_time

        # ---exclusive of the time it takes to move those batches to the device ---
        inputs, labels = inputs.to(device), labels.to(device)

        # --- (C2.2) Training (mini-batch calculation) time ---
        train_start_time = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

        current_train_time = time.perf_counter() - train_start_time
        total_train_time += current_train_time


        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / total:.3f}',
            'Acc': f'{100. * correct / total:.1f}%'
        })

        # reset the timer for next step
        start_data_time = time.perf_counter()

    # --- (C2.3) Total running time ---
    total_running_time = time.perf_counter() - epoch_start_time

    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc, total_data_time, total_train_time, total_running_time
