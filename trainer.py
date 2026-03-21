from collections import Counter
from tqdm.auto import tqdm
import datetime

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from monai.metrics import HausdorffDistanceMetric
from utils.performance import dice_coef
from utils.setting import save_model
from loss import loss_calc

import wandb

def validation(epoch, model, data_loader, loss_list, thr=0.5, classes=None):
    print(f'Start validation #{epoch:2d}')
    model = model.cuda()
    model.eval()

    HD95_metric = HausdorffDistanceMetric(include_background=True, 
                                         distance_metric='euclidean',
                                         percentile=95,
                                         reduction="mean_batch",)
    HD95_metric.reset()

    dices = []
    with torch.no_grad():
        total_loss = 0.0
        total_loss_dict = Counter()
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss, loss_dict = loss_calc(loss_list, outputs, masks, epoch)
            total_loss_dict += Counter(loss_dict)
            total_loss += loss.item()
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().float()
            masks = masks.detach().cpu().float()

            dice = dice_coef(outputs, masks)
            dices.append(dice)
            HD95_metric(outputs, masks)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(classes, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    print(f"Loss: {round(total_loss/cnt, 4)}")
    
    avg_loss_dict = {name: round(val / cnt, 5) for name, val in total_loss_dict.items()}
    print(f"Loss Detail: {avg_loss_dict}")

    avg_dice = torch.mean(dices_per_class).item()

    class_wise_HD95 = HD95_metric.aggregate()
    HD95_metric.reset()

    avg_HD95_dict = {name: round(val.item(), 5) for val, name in zip(class_wise_HD95, classes)}
    print(f"HD95 Detail: {avg_HD95_dict}")

    # wandb에 로그 기록
    log_dict = {
        "val/loss": total_loss / cnt,
        "val/avg dice": avg_dice,
        "epoch": epoch
    }

    for name, val in avg_loss_dict.items():
        log_dict[f"val/loss_{name}"] = val

    for c, d in zip(classes, dices_per_class):
        log_dict[f"val/dice_{c}"] = d.item()

    for name, val in avg_HD95_dict.items():
        log_dict[f"val/HD95_{name}"] = val

    wandb.log(log_dict)

    return avg_dice


scaler = GradScaler()

def train(model, data_loader, val_loader, loss_list, optimizer, scheduler=None, config=None):
    print(f'Start training..')
    model = model.cuda()

    accum_step = config['accum_step']
    num_epochs = config['num_epochs']
    val_every = config['val_every']
    threshold = config['threshold']
    classes = config['classes']
    saved_dir = config['saved_dir']
    saved_name = config['saved_name']
    warmup_epoch = config['warmup_epoch']

    best_dice = 0.
    loss_print_period = len(data_loader)//4

    for epoch in range(1, num_epochs+1):
        model.train()

        if epoch <= warmup_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif epoch == (warmup_epoch + 1):
            for param in model.encoder.parameters():
                param.requires_grad = True

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss, loss_dict = loss_calc(loss_list, outputs, masks, epoch)
                loss = loss / accum_step

            scaler.scale(loss).backward()

            if (step + 1) % accum_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            # step 주기에 따라 loss 출력
            if (step + 1) % loss_print_period == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch}/{num_epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

            current_epoch = (epoch-1) + step / len(data_loader)

            log_dict = {
                "train/loss": loss.item() * accum_step,
                "epoch": current_epoch,
                }
            
            for name, val in loss_dict.items():
                log_dict[f"train/loss_{name}"] = val
            
            wandb.log(log_dict)

            # wandb.log({"train/loss": loss.item() * accum_step,
            #            "epoch": current_epoch})

        # validation 주기에 따라 loss 출력 및 best model 저장
        if epoch % val_every == 0:
            dice = validation(epoch, model, val_loader, loss_list, threshold, classes)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {saved_dir}")
                best_dice = dice
                save_model(model, saved_dir, saved_name)