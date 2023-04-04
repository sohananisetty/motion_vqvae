import torch
import torch.nn as nn

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
            loss = self.Loss(motion_pred[..., : self.motion_dim]*mask[...,None] , motion_gt[..., :self.motion_dim]*mask[...,None]) * (motion_pred.numel()/(mask.sum()*motion_pred.shape[-1]))
            
        return loss
    
    def forward_vel(self, motion_pred, motion_gt , mask = None) : 

        if mask is None:
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])

        else:
            # F.mse_loss(batch["motion"] * batch["motion_mask"][...,None] , pred_motion*batch["motion"] * batch["motion_mask"][...,None], reduction = "sum")
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4]*mask[...,None] , motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4]*mask[...,None]) * (((self.nb_joints - 1) * 3).numel()/(mask.sum()*motion_pred.shape[-1]))
            
        
        return loss