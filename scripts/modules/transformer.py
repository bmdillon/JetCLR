import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# class for transformer network
class Transformer( nn.Module ):
    
    # define and intialize the structure of the neural network
    def __init__( self, input_dim, model_dim, output_dim, n_heads, dim_feedforward, n_layers, learning_rate, n_head_layers=2, head_norm=False, dropout=0.1, opt="adam" ):
        super().__init__()
        # define hyperparameters
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_head_layers = n_head_layers
        self.head_norm = head_norm
        self.dropout = dropout
        # define subnetworks
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout), n_layers)
        # head_layers have output_dim
        if n_head_layers == 0:
            self.head_layers = []
        else:
            if head_norm: self.norm_layers = nn.ModuleList([nn.LayerNorm(model_dim)])
            self.head_layers = nn.ModuleList([nn.Linear(model_dim, output_dim)])
            for i in range(n_head_layers-1):
                if head_norm: self.norm_layers.append(nn.LayerNorm(output_dim))
                self.head_layers.append(nn.Linear(output_dim, output_dim))
        # option to use adam or sgd
        if opt == "adam":
            self.optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate )
        if opt == "sgdca" or opt == "sgdslr" or opt == "sgd":
            self.optimizer = torch.optim.SGD( self.parameters(), lr=self.learning_rate, momentum=0.9 )

    def forward(self, inpt, mask=None, use_mask=False, use_continuous_mask=False, mult_reps=False):
        '''
        input here is (batch_size, n_constit, 3)
        but transformer expects (n_constit, batch_size, 3) so we need to transpose
        if use_mask is True, will mask out all inputs with pT=0
        '''
        assert not (use_mask and use_continuous_mask)
        # make a copy
        x = inpt + 0.   
        # (batch_size, n_constit)
        if use_mask: pT_zero = x[:,:,0] == 0 
        # (batch_size, n_constit)
        if use_continuous_mask: pT = x[:,:,0] 
        if use_mask:
            mask = self.make_mask(pT_zero).to(x.device)
        elif use_continuous_mask:
            mask = self.make_continuous_mask(pT).to(x.device)
        else:
            mask = None
        x = torch.transpose(x, 0, 1)
        # (n_constit, batch_size, model_dim)
        x = self.embedding(x)               
        x = self.transformer(x, mask=mask)
        if use_mask:
            # set masked constituents to zero
            # otherwise the sum will change if the constituents with 0 pT change
            x[torch.transpose(pT_zero, 0, 1)] = 0
        elif use_continuous_mask:
            # scale x by pT, so that function is IR safe
            # transpose first to get correct shape
            x *= torch.transpose(pT, 0, 1)[:,:,None]
        # sum over sequence dim
        # (batch_size, model_dim)
        x = x.sum(0)                        
        return self.head(x, mult_reps)


    def head(self, x, mult_reps):
        '''
        calculates output of the head if it exists, i.e. if n_head_layer>0
        returns multiple representation layers if asked for by mult_reps = True
        input:  x shape=(batchsize, model_dim)
                mult_reps boolean
        output: reps shape=(batchsize, output_dim)                  for mult_reps=False
                reps shape=(batchsize, number_of_reps, output_dim)  for mult_reps=True
        '''
        relu = nn.ReLU()
            # return representations from multiple layers for evaluation
        if mult_reps == True:   
            if self.n_head_layers > 0:
                reps = torch.empty(x.shape[0], self.n_head_layers+1, self.output_dim)
                reps[:, 0] = x
                for i, layer in enumerate(self.head_layers):
                    # only apply layer norm on head if chosen
                    if self.head_norm: x = self.norm_layers[i](x)       
                    x = relu(x)
                    x = layer(x)
                    reps[:, i+1] = x
                # shape (n_head_layers, output_dim)
                return reps  
            # no head exists -> just return x in a list with dimension 1
            else:  
                reps = x[:, None, :]
                # shape (batchsize, 1, model_dim)
                return reps  
        # return only last representation for contrastive loss
        else:  
            for i, layer in enumerate(self.head_layers):  # will do nothing if n_head_layers is 0
                if self.head_norm: x = self.norm_layers[i](x)
                x = relu(x)
                x = layer(x)
            # shape either (model_dim) if no head, or (output_dim) if head exists
            return x  


    def forward_batchwise( self, x, batch_size, use_mask=False, use_continuous_mask=False):
        device = next(self.parameters()).device
        with torch.no_grad():
            if self.n_head_layers == 0:
                rep_dim = self.model_dim
                number_of_reps = 1
            elif self.n_head_layers > 0:
                rep_dim = self.output_dim
                number_of_reps = self.n_head_layers+1
            out = torch.empty( x.size(0), number_of_reps, rep_dim )
            idx_list = torch.split( torch.arange( x.size(0) ), batch_size )
            for idx in idx_list:
                output = self(x[idx].to(device), use_mask=use_mask, use_continuous_mask=use_continuous_mask, mult_reps=True).detach().cpu()
                out[idx] = output
        return out


    def make_mask(self, pT_zero):
        '''
        Input: batch of bools of whether pT=0, shape (batchsize, n_constit)
        Output: mask for transformer model which masks out constituents with pT=0, shape (batchsize*n_transformer_heads, n_constit, n_constit)
        mask is added to attention output before softmax: 0 means value is unchanged, -inf means it will be masked
        '''
        n_constit = pT_zero.size(1)
        pT_zero = torch.repeat_interleave(pT_zero, self.n_heads, axis=0)
        pT_zero = torch.repeat_interleave(pT_zero[:,None], n_constit, axis=1)
        mask = torch.zeros(pT_zero.size(0), n_constit, n_constit)
        mask[pT_zero] = -np.inf
        return mask
    
    
    def make_continuous_mask(self, pT):
        '''
        Input: batch of pT values, shape (batchsize, n_constit)
        Output: mask for transformer model: -1/pT, shape (batchsize*n_transformer_heads, n_constit, n_constit)
        mask is added to attention output before softmax: 0 means value is unchanged, -inf means it will be masked
        intermediate values mean it is partly masked
        This function implements IR safety in the transformer
        '''
        n_constit = pT.size(1)
        pT_reshape = torch.repeat_interleave(pT, self.n_heads, axis=0)
        pT_reshape = torch.repeat_interleave(pT_reshape[:,None], n_constit, axis=1)
        #mask = -1/pT_reshape
        mask = 0.5*torch.log( pT_reshape )
        return mask


