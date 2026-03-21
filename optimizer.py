import torch 
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts, ConstantLR

def get_optimizer(model, optim_config):
    optim_name          = optim_config['name']
    optim_lr            = optim_config['learning_rate']
    optim_weight_decay  = optim_config['weight_decay']

    if optim_name == 'adam':
        optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': float(optim_lr['encoder'])},
        {'params': model.decoder.parameters(), 'lr': float(optim_lr['decoder'])},
        {'params': model.segmentation_head.parameters(), 'lr': float(optim_lr['head'])},
        ],
        weight_decay=float(optim_weight_decay))
        print(f"옵티마이저 설정: Adam, lr={optim_lr}")

    elif optim_name == 'adamW':
        optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': float(optim_lr['encoder'])},
        {'params': model.decoder.parameters(), 'lr': float(optim_lr['decoder'])},
        {'params': model.segmentation_head.parameters(), 'lr': float(optim_lr['head'])},
        ],
        weight_decay=float(optim_weight_decay))
        print(f"옵티마이저 설정: AdamW, lr={optim_lr}")

    else:
        assert "optimizer는 adam, adamW 중 하나를 사용해주시길 바랍니다."

    return optimizer

def get_scheduler(config, optimizer, batch_per_epoch):
    scheduler_list      = []

    target_lr           = config['optimizer']['learning_rate']
    accum_step          = config['train']['gradient_accum']
    scheduler_config    = config['scheduler']
    steps_per_epoch     = batch_per_epoch // accum_step

    warmup_epochs       = scheduler_config['warmup_epoch']
    cosine_period       = scheduler_config['cosine_period']

    warmup_steps        = warmup_epochs * steps_per_epoch
    cosine_steps        = cosine_period * steps_per_epoch

    if scheduler_config['linear']:
        scheduler_list += [LinearLR(optimizer,
                                    start_factor = 0.1,
                                    end_factor = 1.0,
                                    total_iters = warmup_steps)]
        print(f"스케줄러 적용: LinearLR, {warmup_epochs} 에폭")

    if scheduler_config['cosine_restart']:
        scheduler_list += [CosineAnnealingWarmRestarts(optimizer,
                                                    T_0 = cosine_steps,
                                                    eta_min = 1e-5)]
        print(f"스케줄러 적용: Cosine Annealing Warm Restarts {cosine_period} 에폭")  

    if scheduler_config['cosine_anneal']:
        scheduler_list += [CosineAnnealingLR(optimizer,
                                            T_max = cosine_steps,
                                            eta_min = 1e-5)]
        print(f"스케줄러 적용: Cosine Annealing {cosine_period} 에폭")

    if len(scheduler_list) == 1:
        scheduler = scheduler_list[0]
    else:
        scheduler = SequentialLR(
            optimizer,
            schedulers=scheduler_list,
            milestones=[warmup_steps]
        )

    return scheduler