import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss( x_i, x_j, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 )
    z_j = F.normalize( x_j, dim=1 )
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss

def align_loss(x, y, alpha=2):
    xdevice = x.get_device()
    reps_x = x.clone()
    reps_y = y.clone()
    reps_x = F.normalize(reps_x, dim=1).to(xdevice)
    reps_y = F.normalize(reps_y, dim=1).to(xdevice)
    loss_align = (reps_x-reps_y).norm(p=2, dim=1).pow(exponent=alpha).mean()
    return loss_align

def uniform_loss(x, t=2):
    xdevice = x.get_device()
    reps_x = x.clone()
    reps_x = F.normalize(reps_x, dim=1).to(xdevice)
    loss_uniform = torch.pdist(reps_x, p=2).pow(2).mul(-t).exp().mean().log()
    return loss_uniform
    
