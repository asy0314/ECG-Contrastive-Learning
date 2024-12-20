import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import ModelBase


class ModelSimCLR(nn.Module):
    def __init__(self, arch='xresnet1d18', T=0.1, use_mlp=True):
        super(ModelSimCLR, self).__init__()

        self.T = T
        
        # create the encoders
        self.encoder = ModelBase(arch=arch, use_mlp=use_mlp)
    
    def forward(self, im1, im2):
        device = im1.device
        batch_size = im1.shape[0]
        n_views = 2  # currently use only two views (im1 and im2)

        # im1, im2: (B, C, T) each        
        # images: (2B, C, T)
        images = torch.cat([im1, im2], dim=0)

        # features: (2B, num_features)
        features = self.encoder(images) # un-normalized yet

        # labels: (2B,) [0, 1, ..., B-1, 0, 1, ..., B-1]
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        # labels: (2B, 2B) where (i, j) = 1 if i and j is same sample views
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        
        # features: (2B, num_features)
        features = F.normalize(features, dim=1)
        
        # cosine similarity
        # similarity_matrix: (2B, 2B)
        similarity_matrix = torch.matmul(features, features.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        
        # masked labels: (2B, 2B-1) diagonal removed
        labels = labels[~mask].view(labels.shape[0], -1)
        
        # masked labels: (2B, 2B-1) diagonal removed
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # select and combine multiple positives
        # positives: (2B, 1) similarity scores for positive pairs (excluding the self-pairs, which were the diagonal terms)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # select only the negatives the negatives
        # negatives: (2B, 2B-2) -2 since each self and positive pairs were excluded
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # logits: (2B, 2B-1)
        logits = torch.cat([positives, negatives], dim=1)
        # apply temperature
        logits = logits / self.T

        # labels: (2B,) zeros
        # target labels are all zero because we put the positives at the first column in the logits above
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
