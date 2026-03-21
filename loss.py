import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss, FocalLoss, SoftBCEWithLogitsLoss
from monai.losses import HausdorffDTLoss

class ClassWeightedFocalLoss(nn.Module):
    def __init__(self, class_weight=1.0, gamma=2.0):
        super().__init__()
        # 여기서 class_weight는 우리가 아는 focal loss의 alpha와 거의 동
        # (B,C,H,W)에 브로드캐스팅하기 위해서 형태를 (C) -> (1,C,1,1)로 변환
        self.register_buffer("class_weight", class_weight.view(1,-1,1,1))
        self.gamma = gamma

    def forward(self, logits, target):
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none') 
        
        pred = torch.sigmoid(logits)
        p_t = target * pred + (1-target) * (1-pred)
        focal_term = (1-p_t) ** self.gamma

        loss = self.class_weight * focal_term * bce_loss
        return loss.mean()

class ClassWeighted_FocalTversky(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, class_weight=None, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        class_weight = class_weight / class_weight.sum()
        self.register_buffer("class_weight", class_weight)
        self.eps = eps

    def forward(self, logits, target):
        logits = logits.flatten(start_dim=-2) # (B,C,H,W) -> (B,C,H*W)
        target = target.flatten(start_dim=-2) # (B,C,H,W) -> (B,C,H*W)

        pred = torch.sigmoid(logits)

        TP = (pred * target).sum(dim=-1)     # (B,C,H*W) -> (B,C)
        FP = (pred * (1-target)).sum(dim=-1) # (B,C,H*W) -> (B,C)
        FN = ((1-pred) * target).sum(dim=-1) # (B,C,H*W) -> (B,C)
        
        Tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        loss = 1 - Tversky
        loss = loss ** self.gamma 

        if self.class_weight is not None:
            loss = (loss * self.class_weight).sum(dim=-1).mean()    # (B,C) 형태의 loss에 class별 가중치 곱하고 class 차원을 다 더해준다음 배치 단위 평균
        else:
            loss = loss.sum(dim=-1).mean()

        return loss

class ClassWeightedDice(nn.Module):
    def __init__(self, class_weight, eps=1e-6):
        super().__init__()
        class_weight = class_weight / class_weight.sum()
        self.register_buffer("class_weight", class_weight)
        self.eps = eps

    def forward(self, logits, target):
        logits = logits.flatten(start_dim=-2) # (B,C,H,W) -> (B,C,H*W)
        target = target.flatten(start_dim=-2) # (B,C,H,W) -> (B,C,H*W)

        pred = torch.sigmoid(logits)

        intersection = (pred * target).sum(dim=-1)
        union = pred.sum(dim=-1) + target.sum(dim=-1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice

        weighted_dice_loss = (dice_loss * self.class_weight).sum(dim=-1).mean()
        return weighted_dice_loss

class BoundaryWeightedBCE(nn.Module):
    def __init__(self, boundary_weight=5.0):
        super(BoundaryWeightedBCE, self).__init__()
        self.boundary_weight = boundary_weight

    def forward(self, logits, target):
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        max_p = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        min_p = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        boundary = max_p - min_p

        weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary
        weighted_bce = (bce_loss * weight_map).mean()
        
        return weighted_bce

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.register_buffer('kx', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('ky', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def get_edges(self, x):
        channels = x.shape[1]
        device = x.device
        
        kx = self.kx.repeat(channels, 1, 1, 1).to(device)
        ky = self.ky.repeat(channels, 1, 1, 1).to(device)
        
        grad_x = F.conv2d(x, kx, padding=1, groups=channels)
        grad_y = F.conv2d(x, ky, padding=1, groups=channels)
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        pred_edge = self.get_edges(pred)
        target_edge = self.get_edges(target)
        
        return F.mse_loss(pred_edge, target_edge)

def get_loss_list(loss_config):
    loss_list = []

    class_weight = torch.tensor([5.959702, 2.5707648, 1.5488842,
                                 8.719823, 4.3014836, 1.8647279, 1,
                                 7.7523055, 3.4299848, 1.7070067, 1.2162064,
                                 7.785179, 3.9369245, 2.0101676,  1.6831907,
                                 11.571564, 7.020772,   2.9759028, 1.6739168]).cuda()

    if loss_config['bce'] > 0:
        loss_list += [ (nn.BCEWithLogitsLoss(), "BCE", loss_config['bce']) ]
        print("손실 함수 적용: BCEWithLogitsLoss")

    if loss_config['dice'] > 0:
        loss_list += [ (DiceLoss(mode='multilabel', from_logits=True), "DICE", loss_config['dice']) ]
        print("손실 함수 적용: Dice")

    if loss_config['tversky'] > 0:
        loss_list += [ (TverskyLoss(mode='multilabel', from_logits=True, alpha=0.45, beta=0.55), "tversky", loss_config['tversky']) ]
        print("손실 함수 적용: tversky")

    if loss_config['log_dice'] > 0:
        loss_list += [ (DiceLoss(mode='multilabel', log_loss=True, from_logits=True), "Log_DICE", loss_config['log_dice']) ]
        print("손실 함수 적용: Log Dice")

    if loss_config['focal_dice'] > 0:
        loss_list += [ (TverskyLoss(mode='multilabel', from_logits=True, gamma=2.0), "focal_dice", loss_config['focal_dice']) ]
        print("손실 함수 적용: focal_dice")
        
    if loss_config['focal_bce'] > 0:
        loss_list += [ (FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0), "focal_bce", loss_config['focal_bce']) ]
        print("손실 함수 적용: focal_bce")
        
    if loss_config['smooth_bce'] > 0:
        loss_list += [ (SoftBCEWithLogitsLoss(smooth_factor=0.1), "smooth_bce", loss_config['smooth_bce']) ]
        print("손실 함수 적용: smooth_bce")
                
    if loss_config['edge_bce'] > 0:
        loss_list += [ (BoundaryWeightedBCE(), "edge_bce", loss_config['edge_bce']) ]
        print("손실 함수 적용: edge_bce")

    if loss_config['sobel_edge'] > 0:
        loss_list += [ (EdgeLoss(), "sobel_edge", loss_config['sobel_edge']) ]
        print("손실 함수 적용: sobel_edge")
        
    if loss_config['class_weight_dice'] > 0:
        loss_list += [ (ClassWeightedDice(class_weight=class_weight), "class_weight_dice", loss_config['class_weight_dice']) ]
        print("손실 함수 적용: class_weight_dice")

    if loss_config['class_weight_bce'] > 0:
        loss_list += [ (nn.BCEWithLogitsLoss(pos_weight = class_weight.view(len(class_weight),1,1)), "class_weight_bce", loss_config['class_weight_bce']) ]
        print("손실 함수 적용: class_weight_bce")
        
    if loss_config['class_focal_dice'] > 0:
        loss_list += [ (ClassWeighted_FocalTversky(alpha = 0.5, beta = 0.5, gamma = 2.0, class_weight=class_weight), "class_focal_dice", loss_config['class_focal_dice']) ]
        print("손실 함수 적용: class_focal_dice")

    if loss_config['class_focal_bce'] > 0:
        loss_list += [ (ClassWeightedFocalLoss(class_weight=class_weight, gamma=2.0), "class_focal_bce", loss_config['class_focal_bce']) ]
        print("손실 함수 적용: class_focal_bce")
        
    if loss_config['class_focal_tverksy'] > 0:
        loss_list += [ (ClassWeighted_FocalTversky(alpha = 0.3, beta = 0.7, gamma = 2.0, class_weight=class_weight), "class_focal_tverksy", loss_config['class_focal_tverksy']) ]
        print("손실 함수 적용: class_focal_tverksy")

    if loss_config['hausdorff'] > 0:
        loss_list += [ (HausdorffDTLoss(include_background=True, sigmoid=True, reduction="mean"), "hausdorff", loss_config['hausdorff']) ]
        print("손실 함수 적용: hausdorff")
    
    return loss_list

def loss_calc(loss_list, outputs, masks, epoch):
    temp_loss = 0
    loss = 0
    loss_dict = dict()

    for loss_fn, loss_name, weight in loss_list:
        if loss_name == "hausdorff":
            if epoch <= 10:
                continue
            else:
                pass
                # weight = (epoch - 10) / 40 * (1/724) * weight
        temp_loss = loss_fn(outputs, masks) * weight
        loss += temp_loss
        loss_dict[loss_name] = temp_loss.item()

    return loss, loss_dict