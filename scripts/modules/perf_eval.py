# import standard python modules
import os
import sys
import numpy as np
from sklearn import metrics

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import simple FCN network
from modules.fcn_linear import fully_connected_linear_network
from modules.fcn import fully_connected_network

# import preprocessing functions
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

def find_nearest( array, value ):
    array = np.asarray( array )
    idx = ( np.abs( array-value ) ).argmin()
    return array[idx]

def get_perf_stats( labels, measures ):
    measures = np.nan_to_num( measures )
    auc = metrics.roc_auc_score( labels, measures )
    fpr,tpr,thresholds = metrics.roc_curve( labels, measures )
    fpr2 = [ fpr[i] for i in range( len( fpr ) ) if tpr[i]>=0.5]
    tpr2 = [ tpr[i] for i in range( len( tpr ) ) if tpr[i]>=0.5]
    try:
        imtafe = np.nan_to_num( 1 / fpr2[ list( tpr2 ).index( find_nearest( list( tpr2 ), 0.5 ) ) ] )
    except:
        imtafe = 1
    return auc, imtafe

def linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_opt, linear_learning_rate, reps_tr_in, trlab_in, reps_te_in, telab_in ):
    xdevice = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    fcn_linear = fully_connected_linear_network( linear_input_size, 1, linear_opt, linear_learning_rate )
    fcn_linear.to( xdevice )
    bce_loss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    losses = []
    if linear_opt == "sgd":
        scheduler = torch.optim.lr_scheduler.StepLR( fcn_linear.optimizer, 100, gamma=0.6, last_epoch=-1, verbose=False)
    for epoch in range( linear_n_epochs ):
        indices_list = torch.split( torch.randperm( reps_tr_in.shape[0] ), linear_batch_size )
        losses_e = []
        for i, indices in enumerate( indices_list ):
            fcn_linear.optimizer.zero_grad()
            x = reps_tr_in[indices,:]
            l = trlab_in[indices]
            x = torch.Tensor( x ).view( -1, linear_input_size ).to( xdevice )
            l = torch.Tensor( l ).view( -1, 1 ).to( xdevice )
            z = sigmoid( fcn_linear( x ) ).to( xdevice )
            loss = bce_loss( z, l ).to( xdevice )
            loss.backward()
            fcn_linear.optimizer.step()
            losses_e.append( loss.detach().cpu().numpy() )    
        losses.append( np.mean( np.array( losses_e )  ) )
        if linear_opt == "sgd":
            scheduler.step()
    out_dat = fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ).detach().cpu().numpy()
    out_lbs = telab_in
    return out_dat, out_lbs, losses

