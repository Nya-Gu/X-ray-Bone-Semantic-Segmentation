import os
import torch
import numpy as np
import random
import wandb

def save_model(model, saved_dir, saved_name):
    output_path = os.path.join(saved_dir, saved_name)
    torch.save(model, output_path)

def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def wandb_start(config):
    project         = config['wandb']['project']
    notes           = config['wandb']['notes']
    tags            = config['wandb']['tags']
    name            = config['wandb']['name']

    augment_list    = [aug for aug, state in config['augmentation']['train'].items() if state and aug not in ['resize', 'normalize']]
    loss_list       = [loss_name for loss_name, weight in config['loss'].items() if weight > 0]

    config          = {
                        "model":            config['model']['name'],
                        "encoder":          config['model']['encoder'],
                        "pretrained":       config['model']['pretrained'],

                        "epochs":           config['train']['epoch'],
                        "batch_size":       config['train']['batch_size'],
                        "mixed_precision":  config['train']['mixed_precision'],
                        "gradient_accum":   config['train']['gradient_accum'],

                        "augmentation":     augment_list,

                        "optimizer":        config['optimizer']['name'],
                        "learning_rate":    config['optimizer']['learning_rate'],
                        "weight_decay":     config['optimizer']['weight_decay'],

                        "scheduler":        config['scheduler'],
                        "loss":             loss_list,

                        "performance":      config['performance'],
                        "seed":             config['other']['seed'],
                      }
    
    run = wandb.init(project = project,
                     notes = notes,
                     tags = tags,
                     name = name,
                     config = config)
    
    return run

def get_class_index(config):
    target_class = config['data']['target_class']

    if target_class == "all":
        class_index = list(range(29))
    elif target_class == "finger":
        class_index = list(range(19))
    elif target_class == "wrist":
        class_index = list(range(19,27))
    elif target_class == "arm":
        class_index = list(range(27,29))
    else:
        assert "config['data']['target_class']를 finger, wrist, arm 중 하나로 설정해주세요"

    return class_index