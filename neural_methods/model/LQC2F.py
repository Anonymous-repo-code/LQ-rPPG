import sys
import os
import numpy as np

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
    sys.path.append(parent_dir)

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, lecun_normal_
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from neural_methods.loss.NegPearsonLoss import Neg_Pearson


class Label_Encoder(nn.Module):
    def __init__(self, input_seq_length=160, d_model=1, embedding_dim=1, num_blocks=1):
        super().__init__()
        self.dilated_conv_block = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=16, dilation=8),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=32, dilation=16),
        )

        # Mamba Encoder
        self.mamba_encoder = nn.Sequential(*[BiMamba_block(d_model=embedding_dim, bimamba=True) for _ in range(num_blocks)])

        # Positional embedding for explicit time-axis information
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq_length, embedding_dim))

        # Weight initialization
        self.apply(segm_init_weights)


    def forward(self, x):
        residual = x

        x = self.dilated_conv_block(x)

        x = x + residual

        x = x.permute(0, 2, 1)

        x = x + self.position_embedding[:, :x.size(1), :] 

        x = self.mamba_encoder(x) # for global time-axis feature extraction

        return x 


class Quantizer(nn.Module):
    def __init__(self, num_codebooks=12, embedding_dim=1, commitment_cost=0.5, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # self.latent_dropout = LatentDropout(dropout_prob=0.1)
        self.embeddings = nn.Embedding(self.num_codebooks, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_codebooks, 1 / self.num_codebooks)

        # EMA parameters
        self.register_buffer('ema_cluster_size', torch.zeros(num_codebooks))
        self.ema_w = nn.Parameter(torch.Tensor(num_codebooks, embedding_dim))
        self.ema_w.data.uniform_(-1 / self.num_codebooks, 1 / self.num_codebooks)

        # Weight initialization
        self.apply(segm_init_weights)

    def forward(self, embedding_tokens):
        # inputs: (B, num_tokens, embedding_dim)
        embedding_tokens = embedding_tokens.contiguous()  # (B, num_tokens, embedding_dim)
        input_shape = embedding_tokens.shape

        flat_inputs = embedding_tokens.view(-1, self.embedding_dim)  # (B*num_tokens, embedding_dim)

        distances = torch.cdist(flat_inputs, self.embeddings.weight) 
        encoding_index = torch.argmin(distances, dim=1) 
        
        quantized = torch.index_select(self.embeddings.weight, 0, encoding_index).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), embedding_tokens)  
        c_loss = self.commitment_cost * e_latent_loss  

        # Forward: quantized
        # Backward: embedding_tokens (i.e., gradient of quantized is directly propagated to embedding_tokens)
        quantized = embedding_tokens + (quantized - embedding_tokens).detach() 
        quantized = quantized.contiguous()

        # Restore batch dimension of encoding_index: (B, num_tokens)
        encoding_index = encoding_index.view(input_shape[0], input_shape[1])

        # EMA update (in-place update)
        if self.training: # Do not update in eval mode, otherwise it will update during validation. EMA-based update is not learning-based
            with torch.no_grad():
                # Flatten encoding indices and apply one-hot encoding
                encoding_index_flat = encoding_index.view(-1)  # Flatten to 1D
                encoding_one_hot = F.one_hot(encoding_index_flat, self.num_codebooks).type_as(flat_inputs)

                # Reshape encoding_one_hot to 2D
                encoding_one_hot = encoding_one_hot.view(input_shape[0] * input_shape[1], -1)
                
                # Check how many times each code is selected
                new_cluster_size = torch.sum(encoding_one_hot, dim=0)
                # in-place update of ema_cluster_size
                self.ema_cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=(1 - self.decay))

                # Laplace smoothing
                n = torch.sum(self.ema_cluster_size)
                smoothed_cluster_size = ((self.ema_cluster_size + self.epsilon) /
                                        (n + self.num_codebooks * self.epsilon) * n)
                self.ema_cluster_size.copy_(smoothed_cluster_size)

                dw = torch.matmul(encoding_one_hot.t(), flat_inputs)
                # in-place update of ema_w
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

                # Update embeddings.weight based on updated ema_w and ema_cluster_size
                self.embeddings.weight.data.copy_(self.ema_w.data / self.ema_cluster_size.unsqueeze(1))


        return c_loss, quantized, encoding_index

    
class Label_Quantizer(nn.Module):
    def __init__(self, num_codebooks=16):
        super().__init__()
        
        self.encoder = Label_Encoder()

        self.quantizer = Quantizer(num_codebooks=num_codebooks)

        self.neg_loss = Neg_Pearson()

        # Weight initialization
        self.apply(segm_init_weights)

    def forward(self, inputs):
        # inputs: (B, T)
        inputs = inputs.unsqueeze(1)  # (B, 1, T)
        x = self.encoder(inputs)
        c_loss, quantized, embedding_indices = self.quantizer(x)
        quantized = quantized.squeeze(-1) # (B, T)
        inputs = inputs.squeeze(1)
        # rec_loss = F.mse_loss(quantized, inputs)
        return c_loss.unsqueeze(0), quantized, embedding_indices
        

class Q2FPhys_backbone(nn.Module):
    def __init__(self,  input_length=160, num_tokens=160, stem_embedding_dim=64, num_blocks=1):
        super().__init__()

        self.stem_embedding_dim = stem_embedding_dim
        self.use_soft_recon = True
        
        self.Fusion_Stem = Fusion_Stem(dim=self.stem_embedding_dim//4)
        self.attn_mask = Attention_mask()

        # Stem3에 BatchNorm 추가
        self.stem3 = nn.Sequential(
            nn.Conv3d(self.stem_embedding_dim//4, self.stem_embedding_dim, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1,2,2)),
            nn.BatchNorm3d(self.stem_embedding_dim),
        )

        self.mamba_1 = nn.Sequential(*[BiMamba_block(d_model=self.stem_embedding_dim) for _ in range(num_blocks)])
        self.mamba_2 = nn.Sequential(*[BiMamba_block(d_model=self.stem_embedding_dim) for _ in range(num_blocks)])
        self.mamba_3 = nn.Sequential(*[BiMamba_block(d_model=self.stem_embedding_dim) for _ in range(num_blocks)])
        self.mamba_4 = nn.Sequential(*[BiMamba_block(d_model=self.stem_embedding_dim) for _ in range(num_blocks)])

        self.mamba1_token_head = nn.Conv1d(stem_embedding_dim, 1, kernel_size=1, stride=1, padding=0)
        self.mamba2_token_head = nn.Conv1d(stem_embedding_dim, 1, kernel_size=1, stride=1, padding=0)
        self.mamba3_token_head = nn.Conv1d(stem_embedding_dim, 1, kernel_size=1, stride=1, padding=0)
        self.mamba4_token_head = nn.Conv1d(stem_embedding_dim, 1, kernel_size=1, stride=1, padding=0)

        # Scalar projection layers for soft reconstruction (1 -> stem_embedding_dim)
        self.mamba1_scalar_projection = nn.Conv1d(1, stem_embedding_dim, kernel_size=1, stride=1, padding=0)
        self.mamba2_scalar_projection = nn.Conv1d(1, stem_embedding_dim, kernel_size=1, stride=1, padding=0)
        self.mamba3_scalar_projection = nn.Conv1d(1, stem_embedding_dim, kernel_size=1, stride=1, padding=0)
        self.mamba4_scalar_projection = nn.Conv1d(1, stem_embedding_dim, kernel_size=1, stride=1, padding=0)

        # Placeholders for pretrained codebooks (shape: (K, 1))
        self.codebook_1 = None
        self.codebook_2 = None
        self.codebook_3 = None
        self.codebook_4 = None

        # Positional embedding for explicit time-axis information
        self.position_embedding = nn.Parameter(torch.zeros(1, input_length, self.stem_embedding_dim))
       
        # Weight initialization
        self.apply(segm_init_weights)

    def load_codebooks(self, codebook_1=None, codebook_2=None, codebook_3=None, codebook_4=None):
        """Load pretrained codebooks for soft reconstruction.
        Args:
            codebook_1: Tensor of shape (2, 1)
            codebook_2: Tensor of shape (4, 1)
            codebook_3: Tensor of shape (8, 1)
            codebook_4: Tensor of shape (16, 1)
        """
        if codebook_1 is not None:
            self.codebook_1 = codebook_1.detach().clone()
        if codebook_2 is not None:
            self.codebook_2 = codebook_2.detach().clone()
        if codebook_3 is not None:
            self.codebook_3 = codebook_3.detach().clone()
        if codebook_4 is not None:
            self.codebook_4 = codebook_4.detach().clone()

    def forward(self, x):
        # ==================================== STEM by RhythmMamba ====================================
        
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape

        x = self.Fusion_Stem(x)
        x = x.view(B, T, self.stem_embedding_dim//4, H//8, W//8).permute(0,2,1,3,4)
        x = self.stem3(x) 
        
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask 

        x = torch.mean(x, 4)
        x = torch.mean(x, 3)  
        x = rearrange(x, 'b c t -> b t c') 
        
        # ==================================== STEM by RhythmMamba ====================================

        # x = self.conv_block(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x = self.dilated_conv_block(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        x = x + self.position_embedding[:, :x.size(1), :]

        mamba1_output = self.mamba_1(x) 
        mamba1_token_head = self.mamba1_token_head(mamba1_output.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, 1)
        if self.use_soft_recon and (self.codebook_1 is not None):
            # distance-based soft reconstruction
            codebook_1 = self.codebook_1.to(mamba1_token_head.device, dtype=mamba1_token_head.dtype)  # (K1, 1)
            logits_1 = -torch.cdist(mamba1_token_head.reshape(-1, 1), codebook_1)  # (B*T, K1)
            probs_1 = F.softmax(logits_1, dim=-1)
            soft_recon_1 = torch.matmul(probs_1, codebook_1).view(mamba1_token_head.size(0), 1, mamba1_token_head.size(1))  # (B, 1, T)
            mamba1_projection = self.mamba1_scalar_projection(soft_recon_1).permute(0, 2, 1)  # (B, T, C)
        else:
            mamba1_projection = self.mamba1_scalar_projection(mamba1_token_head.permute(0, 2, 1)).permute(0, 2, 1)
        mamba1_output = mamba1_output + mamba1_projection
        
        

        mamba2_output = self.mamba_2(mamba1_output) 
        mamba2_token_head = self.mamba2_token_head(mamba2_output.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, 1)
        if self.use_soft_recon and (self.codebook_2 is not None):
            codebook_2 = self.codebook_2.to(mamba2_token_head.device, dtype=mamba2_token_head.dtype)  # (K2, 1)
            logits_2 = -torch.cdist(mamba2_token_head.reshape(-1, 1), codebook_2)
            probs_2 = F.softmax(logits_2, dim=-1)
            soft_recon_2 = torch.matmul(probs_2, codebook_2).view(mamba2_token_head.size(0), 1, mamba2_token_head.size(1))
            mamba2_projection = self.mamba2_scalar_projection(soft_recon_2).permute(0, 2, 1)
        else:
            mamba2_projection = self.mamba2_scalar_projection(mamba2_token_head.permute(0, 2, 1)).permute(0, 2, 1)
        mamba2_output = mamba2_output + mamba2_projection  + mamba1_projection

        mamba3_output = self.mamba_3(mamba2_output) 
        mamba3_token_head = self.mamba3_token_head(mamba3_output.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, 1)
        if self.use_soft_recon and (self.codebook_3 is not None):
            codebook_3 = self.codebook_3.to(mamba3_token_head.device, dtype=mamba3_token_head.dtype)  # (K3, 1)
            logits_3 = -torch.cdist(mamba3_token_head.reshape(-1, 1), codebook_3)
            probs_3 = F.softmax(logits_3, dim=-1)
            soft_recon_3 = torch.matmul(probs_3, codebook_3).view(mamba3_token_head.size(0), 1, mamba3_token_head.size(1))
            mamba3_projection = self.mamba3_scalar_projection(soft_recon_3).permute(0, 2, 1)
        else:
            mamba3_projection = self.mamba3_scalar_projection(mamba3_token_head.permute(0, 2, 1)).permute(0, 2, 1) 
        mamba3_output = mamba3_output + mamba3_projection  + mamba2_projection + mamba1_projection

        mamba4_output = self.mamba_4(mamba3_output) 
        mamba4_token_head = self.mamba4_token_head(mamba4_output.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, 1)
        if self.use_soft_recon and (self.codebook_4 is not None):
            codebook_4 = self.codebook_4.to(mamba4_token_head.device, dtype=mamba4_token_head.dtype)  # (K4, 1)
            logits_4 = -torch.cdist(mamba4_token_head.reshape(-1, 1), codebook_4)
            probs_4 = F.softmax(logits_4, dim=-1)
            soft_recon_4 = torch.matmul(probs_4, codebook_4).view(mamba4_token_head.size(0), 1, mamba4_token_head.size(1))
            mamba4_projection = self.mamba4_scalar_projection(soft_recon_4).permute(0, 2, 1)
        else:
            mamba4_projection = self.mamba4_scalar_projection(mamba4_token_head.permute(0, 2, 1)).permute(0, 2, 1) 
        mamba4_output = mamba4_output + mamba4_projection  + mamba3_projection + mamba2_projection + mamba1_projection

        return mamba4_output, mamba1_token_head, mamba2_token_head, mamba3_token_head, mamba4_token_head
    

class Q2FPhys_rPPG_predictor(nn.Module):
    def __init__(self, num_tokens=160, stem_embedding_dim=64, num_codebooks=12):
        super().__init__()

        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(stem_embedding_dim, stem_embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(stem_embedding_dim, stem_embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(stem_embedding_dim, stem_embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        self.last_conv = nn.Conv1d(stem_embedding_dim, 1, kernel_size=3, stride=1, padding=1)

        self.apply(segm_init_weights)

    def forward(self, mamba4_output):
        x = mamba4_output.permute(0, 2, 1) # (B, stem_embedding_dim, T)
        x = self.time_feature_extractor(x) # (B, stem_embedding_dim, T)
        x = self.last_conv(x) # (B, 1, T)
        
        final_token_head = x.permute(0, 2, 1) # (B, T, 1)

        return final_token_head.squeeze(-1) # (B, T)


class C2F_model(Q2FPhys_backbone):
    def __init__(self, input_length=160, num_tokens=160, stem_embedding_dim=64, num_blocks=1):
        super().__init__(input_length=input_length, num_tokens=num_tokens, stem_embedding_dim=stem_embedding_dim, num_blocks=num_blocks)
        self.rppg_predictor = Q2FPhys_rPPG_predictor(stem_embedding_dim=stem_embedding_dim)

    def forward(self, x):
        mamba4_output, mamba1_token_head, mamba2_token_head, mamba3_token_head, mamba4_token_head = super().forward(x)
        final_token_head = self.rppg_predictor(mamba4_output)
        return final_token_head, mamba1_token_head, mamba2_token_head, mamba3_token_head, mamba4_token_head


class BiMamba_block(nn.Module):
    def __init__(self, d_model=16, d_state=48, d_conv=4, expand=2, bimamba=True):
        super().__init__()

        self.x_norm = nn.LayerNorm(d_model)

        self.mamba = Mamba(d_model=d_model,
                           d_state=d_state,
                           d_conv=d_conv,
                           expand=expand,
                           bimamba=True)
    
        self.apply(segm_init_weights)

    def forward(self, x):
        residual = x
        x = self.x_norm(x)
        x = self.mamba(x)
        x = x + residual
        return x


class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):
        super(Fusion_Stem, self).__init__()


        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,dim,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2

        return x

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config
    

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
