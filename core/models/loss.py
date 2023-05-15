import torch
import torch.nn as nn
import torch.nn.functional as F

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt , mask = None) : 
        ## pred: b n d, gt: b n d, mask: b n
        if mask is None:
            loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        else:
            # F.mse_loss(batch["motion"] * batch["motion_mask"][...,None] , pred_motion*batch["motion"] * batch["motion_mask"][...,None], reduction = "sum")
            norm =  motion_pred.numel()/(mask.sum()*motion_pred.shape[-1])
            loss = self.Loss(motion_pred[..., : self.motion_dim]*mask[...,None] , motion_gt[..., :self.motion_dim]*mask[...,None]) * norm
            
        return loss
    
    def forward_vel(self, motion_pred, motion_gt , mask = None) : 

        if mask is None:
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])

        else:
            norm = motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4].numel()/(mask.sum()*motion_pred.shape[-1])
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4]*mask[...,None] , motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4]*mask[...,None]) * norm
            
        
        return loss
    
    
    
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
  
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
