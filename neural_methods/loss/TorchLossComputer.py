'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
from evaluation.post_process import calculate_hr , calculate_psd

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss

class Hybrid_Loss(nn.Module): 
    def __init__(self):
        super(Hybrid_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_Fre , _ = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_Fre
        return loss

class Hybrid_Loss_Batched(nn.Module):
    def __init__(self):
        super(Hybrid_Loss_Batched, self).__init__()

    @staticmethod
    def _batched_pearson_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        preds, labels: (B, T) -> mean(1 - pearson) over batch
        """
        preds = preds.float()
        labels = labels.float()
        preds = preds - preds.mean(dim=1, keepdim=True)
        labels = labels - labels.mean(dim=1, keepdim=True)
        numerator = (preds * labels).sum(dim=1)
        denom = torch.sqrt((preds.pow(2).sum(dim=1)) * (labels.pow(2).sum(dim=1)) + 1e-8)
        pearson = numerator / (denom + 1e-8)
        return (1.0 - pearson).mean()

    def forward(self, pred_ppg: torch.Tensor, labels: torch.Tensor, epoch, FS, diff_flag):
        """
        pred_ppg: (B, T) or (B, T, 1)
        labels:   (B, T) or (B, T, 1)
        """
        if pred_ppg.dim() == 3 and pred_ppg.size(-1) == 1:
            pred_ppg = pred_ppg.squeeze(-1)
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)

        loss_time = self._batched_pearson_loss(pred_ppg, labels)
        ce_loss, _ = TorchLossComputer.Frequency_loss_batched(pred_ppg, labels, diff_flag=diff_flag, Fs=FS, std=3.0)
        loss = 0.2 * loss_time + 1.0 * ce_loss
        return loss
    
class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def compute_complex_absolute_given_k_batched(output, k, N):
        """
        Batched version.
        Args:
            output: (B, T)
            k: (K,) tensor of frequency indices
            N: int (T)
        Returns:
            (B, K) power per frequency index
        """
        device = output.device
        dtype = torch.float32

        two_pi_n_over_N = (2 * math.pi * torch.arange(0, N, dtype=dtype, device=device)) / N
        hanning = torch.from_numpy(np.hanning(N)).type(dtype).to(device).view(1, -1)

        # Windowing
        output = (output * hanning)  # (B, T)
        output = output.view(output.shape[0], 1, -1)  # (B, 1, T)

        k = k.type(dtype).to(device).view(1, -1, 1)  # (1, K, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)  # (1, 1, T)

        sin_term = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1)
        cos_term = torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1)
        complex_absolute = sin_term ** 2 + cos_term ** 2  # (B, K)

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
    
    @staticmethod
    def complex_absolute_batched(output, Fs, bpm_range=None):
        """
        Batched PSD-like energy at discrete bpm indices.
        Args:
            output: (B, T)
            Fs: sampling rate
            bpm_range: (K,) tensor of integer bpm values
        Returns:
            (B, K) normalized energy per bpm index
        """
        B, T = output.shape
        N = T
        unit_per_hz = Fs / N
        feasible_bpm = bpm_range.to(output.device) / 60.0
        k = feasible_bpm / unit_per_hz  # (K,)
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k_batched(output, k, N)  # (B, K)
        return (1.0 / complex_absolute.sum(dim=1, keepdim=True)) * complex_absolute
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(torch.device('cuda'))
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)

    @staticmethod
    def Frequency_loss_batched(inputs, target, diff_flag, Fs, std):
        """
        Batched frequency-domain loss.
        Args:
            inputs: (B, T)
            target: (B, T)
        Returns:
            ce_mean, kl_mean (both scalars)
        """
        device = inputs.device
        bpm_range = torch.arange(45, 150, dtype=torch.float, device=device)

        # Predicted energy distribution over bpm (B, K)
        ca_pred = TorchLossComputer.complex_absolute_batched(inputs, Fs, bpm_range)
        sa_pred = ca_pred / torch.sum(ca_pred, dim=1, keepdim=True)  # (B, K)

        # Target HR index via argmax over target's energy distribution
        ca_tgt = TorchLossComputer.complex_absolute_batched(target, Fs, bpm_range)  # (B, K)
        hr_idx = torch.argmax(ca_tgt, dim=1)  # (B,)

        # CE loss over batch
        ce_loss = F.cross_entropy(ca_pred, hr_idx)

        # KL loss to Gaussian around HR (vectorized)
        # Build Gaussian distributions centered at bpm = hr_idx + 45
        mu = (hr_idx + 45).unsqueeze(1).float()  # (B,1)
        k_grid = bpm_range.view(1, -1)  # (1, K)
        gaussian = torch.exp(-((k_grid - mu) ** 2) / (2 * (std ** 2))) / (math.sqrt(2 * math.pi) * std)
        gaussian = torch.clamp(gaussian, min=1e-15)
        gaussian = gaussian / gaussian.sum(dim=1, keepdim=True)

        # Use log for KLDivLoss-like computation: sum p*log(p/q)
        kl = torch.sum(gaussian * (torch.log(gaussian) - torch.log(torch.clamp(sa_pred, min=1e-15))), dim=1)
        kl_loss_mean = kl.mean()

        return ce_loss, kl_loss_mean
    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        return kl_loss(pred_distribution , target_distribution)