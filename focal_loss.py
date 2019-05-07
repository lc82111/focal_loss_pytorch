import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FocalLoss_v2(nn.Module):
    def __init__(self, num_class, gamma=2, alpha=None):
        '''
        alpha: tensor of shape (C)
        '''
        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha==None:
            self.alpha = torch.ones(num_class)
        if isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class)
            self.alpha = alpha / alpha.sum()

    def forward(self, logit, target):
        '''
        args: logits: tensor before the softmax of shape (N,C) where C = number of classes 
            or (N, C, H, W) in case of 2D Loss, 
            or (N,C,d1,d2,...,dK) where K≥1 in the case of K-dimensional loss.
        args: label: (N) where each value is in [0,C-1],
            or (N, d1, d2, ..., dK)
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
        '''
        if self.alpha.device != logit.device:
            self.alpha = self.alpha.to(logit.device)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1) #(N,C,H*W)
            logit = logit.permute(0, 2, 1).contiguous() #(N,H*W,C)
            logit = logit.view(-1, logit.size(-1)) #(N*H*W,C)
        target = target.view(-1) #(N*H*W)
        #alpha  = self.alpha.view(1, self.num_class) #(1,C)
        alpha = self.alpha[target.cpu().long()] #(N*H*W)

        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt    = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()
        
def test_focal():
    num_class = 5

    nodes = 100
    N = 100
    # model1d = torch.nn.Linear(nodes, num_class).cuda()
    model2d = torch.nn.Conv2d(16, num_class, 3, padding=1).cuda()
    alpha = [0.1,0.1,0.1,0.2,0.5]
    FL2 = FocalLoss_v2(num_class=num_class, alpha=alpha)
    for i in range(10):
        input  = torch.rand(3, 16, 32, 32).cuda() #(B,C,H,W)
        target = torch.rand(3, 32, 32).random_(num_class).cuda() #(B,H,W)
        target = target.long().cuda()
        output = model2d(input) #(B,num_classes,H,W)
        loss2 = FL2(output, target)
        print(loss2.item())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_focal()
