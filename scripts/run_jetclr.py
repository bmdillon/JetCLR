#!/bin/env python3.7

# load standard python modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules required for jetCLR training
from modules.jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets
from modules.transformer import Transformer
from modules.losses import contrastive_loss, align_loss, uniform_loss
from modules.perf_eval import get_perf_stats, linear_classifier_test 

# import args from extargs.py file
import extargs as args

# set the number of threads that pytorch will use
torch.set_num_threads(2)

t0 = time.time()

# initialise logfile
logfile = open( args.logfile, "a" )
print( "logfile initialised", file=logfile, flush=True )

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device ), flush=True, file=logfile )

# set up results directory
base_dir = "/your/project/directory/"
expt_tag = args.expt
expt_dir = base_dir + "experiments/" + expt_tag + "/"

# check if experiment already exists
if os.path.isdir(expt_dir):
    sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")
else:
    os.makedirs(expt_dir)
print("experiment: "+str(args.expt), file=logfile, flush=True)

# load data
print( "loading data", flush=True, file=logfile )
tr_dat_in = np.load( args.tr_dat_path )
tr_lab_in = np.load( args.tr_lab_path )

# input dim to the transformer -> (pt,eta,phi)
input_dim = 3

# creating the training dataset
print( "shuffling data and doing the S/B split", flush=True, file=logfile )
tr_bkg_dat = tr_dat_in[ tr_lab_in==0 ].copy()
tr_sig_dat = tr_dat_in[ tr_lab_in==1 ].copy()
nbkg_tr = int( tr_bkg_dat.shape[0] )
nsig_tr = int( args.sbratio * nbkg_tr )
list_tr_dat = list( tr_bkg_dat[ 0:nbkg_tr ] ) + list( tr_sig_dat[ 0:nsig_tr ] )
list_tr_lab = [ 0 for i in range( nbkg_tr ) ] + [ 1 for i in range( nsig_tr ) ]
ldz_tr = list( zip( list_tr_dat, list_tr_lab ) )
random.shuffle( ldz_tr )
tr_dat, tr_lab = zip( *ldz_tr )
# reducing the training data
tr_dat = np.array( tr_dat )[0:100000]
tr_lab = np.array( tr_lab )[0:100000]

# create two validation sets: 
# one for training the linear classifier test (LCT)
# and one for testing on it
# we will do this just with tr_dat_in, but shuffled and split 50/50
# this should be fine because the jetCLR training doesn't use labels
# we want the LCT to use S/B=1 all the time
list_vl_dat = list( tr_dat_in.copy() )
list_vl_lab = list( tr_lab_in.copy() )
ldz_vl = list( zip( list_vl_dat, list_vl_lab ) )
random.shuffle( ldz_vl )
vl_dat, vl_lab = zip( *ldz_vl )
vl_dat = np.array( vl_dat )
vl_lab = np.array( vl_lab )
vl_len = vl_dat.shape[0]
vl_split_len = int( vl_len/2 )
vl_dat_1 = vl_dat[ 0:vl_split_len ]
vl_lab_1 = vl_lab[ 0:vl_split_len ]
vl_dat_2 = vl_dat[ -vl_split_len: ]
vl_lab_2 = vl_lab[ -vl_split_len: ]

# cropping all jets to a fixed number of consituents
tr_dat = crop_jets( tr_dat, args.nconstit )
vl_dat_1 = crop_jets( vl_dat_1, args.nconstit )
vl_dat_2 = crop_jets( vl_dat_2, args.nconstit )

# print data dimensions
print( "training data shape: " + str( tr_dat.shape ), flush=True, file=logfile )
print( "validation-1 data shape: " + str( vl_dat_1.shape ), flush=True, file=logfile )
print( "validation-2 data shape: " + str( vl_dat_2.shape ), flush=True, file=logfile )
print( "training labels shape: " + str( tr_lab.shape ), flush=True, file=logfile )
print( "validation-1 labels shape: " + str( vl_lab_1.shape ), flush=True, file=logfile )
print( "validation-2 labels shape: " + str( vl_lab_2.shape ), flush=True, file=logfile )

t1 = time.time()

# re-scale test data, for the training data this will be done on the fly due to the augmentations
vl_dat_1 = rescale_pts( vl_dat_1 )
vl_dat_2 = rescale_pts( vl_dat_2 )

print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds", flush=True, file=logfile )

# set-up parameters for the LCT
linear_input_size = args.output_dim
linear_n_epochs = 750
linear_learning_rate = 0.001
linear_batch_size = 128

print( "--- contrastive learning transformer network architecture ---", flush=True, file=logfile )
print( "model dimension: " + str( args.model_dim ) , flush=True, file=logfile )
print( "number of heads: " + str( args.n_heads ) , flush=True, file=logfile )
print( "dimension of feedforward network: " + str( args.dim_feedforward ) , flush=True, file=logfile )
print( "number of layers: " + str( args.n_layers ) , flush=True, file=logfile )
print( "number of head layers: " + str( args.n_head_layers ) , flush=True, file=logfile )
print( "optimiser: " + str( args.opt ) , flush=True, file=logfile )
print( "mask: " + str( args.mask ) , flush=True, file=logfile )
print( "continuous mask: " + str( args.cmask ) , flush=True, file=logfile )
print( "--- hyper-parameters ---", flush=True, file=logfile )
print( "learning rate: " + str( args.learning_rate ) , flush=True, file=logfile )
print( "batch size: " + str( args.batch_size ) , flush=True, file=logfile )
print( "temperature: " + str( args.temperature ) , flush=True, file=logfile )
print( "--- symmetries/augmentations ---", flush=True, file=logfile )
print( "rotations: " + str( args.rot ) , flush=True, file=logfile )
print( "low pT smearing: " + str( args.ptd ) , flush=True, file=logfile )
print( "pT smearing clip parameter: " + str( args.ptcm ) , flush=True, file=logfile )
print( "translations: " + str( args.trs ) , flush=True, file=logfile )
print( "translations width: " + str( args.trsw ) , flush=True, file=logfile )
print( "---", flush=True, file=logfile )

# initialise the network
print( "initialising the network", flush=True, file=logfile )
net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )

# send network to device
net.to( device )

# set learning rate scheduling, if required
# SGD with cosine annealing
if args.opt == "sgdca":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts( net.optimizer, 15, T_mult=2, eta_min=0, last_epoch=-1, verbose=False)
# SGD with step-reduced learning rates
if args.opt == "sgdslr":
    scheduler = torch.optim.lr_scheduler.StepLR( net.optimizer, 100, gamma=0.6, last_epoch=-1, verbose=False)

# THE TRAINING LOOP

print( "starting training loop, running for " + str( args.n_epochs ) + " epochs", flush=True, file=logfile )
print( "---", flush=True, file=logfile )

# initialise lists for storing training stats
auc_epochs = []
imtafe_epochs = []
losses = []
loss_align_epochs = []
loss_uniform_epochs = []

# cosine annealing requires per-batch calls to the scheduler, we need to know the number of batches per epoch
if args.opt == "sgdca":
    # number of iterations per epoch
    iters = int( tr_dat.shape[0]/args.batch_size )
    print( "number of iterations per epoch: " + str(iters), flush=True, file=logfile )

# the loop
for epoch in range( args.n_epochs ):

    # re-batch the data on each epoch
    indices_list = torch.split( torch.randperm( tr_dat.shape[0] ), args.batch_size )

    # initialise timing stats
    te0 = time.time()
    
    # initialise lists to store batch stats
    loss_align_e = []
    loss_uniform_e = []
    losses_e = []

    # initialise timing stats
    td1 = 0
    td2 = 0
    td3 = 0
    td4 = 0
    td5 = 0
    td6 = 0
    td7 = 0
    td8 = 0

    # the inner loop goes through the dataset batch by batch
    # augmentations of the jets are done on the fly
    for i, indices in enumerate( indices_list ):
        net.optimizer.zero_grad()
        x_i = tr_dat[indices,:,:]
        time1 = time.time()
        x_i = rotate_jets( x_i )
        x_j = x_i.copy()
        if args.rot:
            x_j = rotate_jets( x_j )
        time2 = time.time()
        if args.cf:
            x_j = collinear_fill_jets( x_j )
            x_j = collinear_fill_jets( x_j )
        time3 = time.time()
        if args.ptd:
            x_j = distort_jets( x_j, strength=args.ptst, pT_clip_min=args.ptcm )
        time4 = time.time()
        if args.trs:
            x_j = translate_jets( x_j, width=args.trsw )
            x_i = translate_jets( x_i, width=args.trsw )
        time5 = time.time()
        x_i = rescale_pts( x_i )
        x_j = rescale_pts( x_j )
        x_i = torch.Tensor( x_i ).transpose(1,2).to( device )
        x_j = torch.Tensor( x_j ).transpose(1,2).to( device )
        time6 = time.time()
        z_i = net( x_i, use_mask=args.mask, use_continuous_mask=args.cmask )
        z_j = net( x_j, use_mask=args.mask, use_continuous_mask=args.cmask )
        time7 = time.time()

        # calculate the alignment and uniformity loss for each batch
        if epoch%10==0:
            loss_align = align_loss( z_i, z_j )
            loss_uniform_zi = uniform_loss( z_i )
            loss_uniform_zj = uniform_loss( z_j )
            loss_align_e.append( loss_align.detach().cpu().numpy() )
            loss_uniform_e.append( ( loss_uniform_zi.detach().cpu().numpy() + loss_uniform_zj.detach().cpu().numpy() )/2 )
        time8 = time.time()

        # compute the loss, back-propagate, and update scheduler if required
        loss = contrastive_loss( z_i, z_j, args.temperature ).to( device )
        loss.backward()
        net.optimizer.step()
        if args.opt == "sgdca":
            scheduler.step( epoch + i / iters )
        losses_e.append( loss.detach().cpu().numpy() )
        time9 = time.time()

        # update timiing stats
        td1 += time2 - time1
        td2 += time3 - time2
        td3 += time4 - time3
        td4 += time5 - time4
        td5 += time6 - time5
        td6 += time7 - time6
        td7 += time8 - time7
        td8 += time9 - time8

    loss_e = np.mean( np.array( losses_e ) )
    losses.append( loss_e )

    if args.opt == "sgdslr":
        scheduler.step()

    te1 = time.time()

    print( "epoch: " + str( epoch ) + ", loss: " + str( round(losses[-1], 5) ), flush=True, file=logfile )
    if args.opt == "sgdca" or args.opt == "sgdslr":
        print( "lr: " + str( scheduler._last_lr ), flush=True, file=logfile )
    print( f"total time taken: {round( te1-te0, 1 )}s, augmentation: {round(td1+td2+td3+td4+td5,1)}s, forward {round(td6, 1)}s, backward {round(td8, 1)}s, other {round(te1-te0-(td1+td2+td3+td4+td6+td7+td8), 2)}s", flush=True, file=logfile )

    # check memory stats on the gpu
    if epoch % 10 == 0:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print( f"CUDA memory: total {t}, reserved {r}, allocated {a}", flush=True, file=logfile )

    # summarise alignment and uniformity stats
    if epoch%10==0:
        loss_align_epochs.append( np.mean( np.array( loss_align_e ) ) )
        loss_uniform_epochs.append( np.mean( np.array( loss_uniform_e ) ) )
        print( "alignment: " + str( loss_align_epochs[-1] ) + ", uniformity: " + str( loss_uniform_epochs[-1] ), flush=True, file=logfile )

    # check number of threads being used
    if epoch%10==0:
        print( "num threads in use: " + str( torch.get_num_threads() ), flush=True, file=logfile )

    # run a short LCT
    if epoch%10==0:
        print( "--- LCT ----" , flush=True, file=logfile )
        if args.trs:
            vl_dat_1 = translate_jets( vl_dat_1, width=args.trsw )
            vl_dat_2 = translate_jets( vl_dat_2, width=args.trsw )
        # get the validation reps
        with torch.no_grad():
            net.eval()
            #vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
            #vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
            vl_reps_1 = net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
            vl_reps_2 = net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
            net.train()
        # running the LCT on each rep layer
        auc_list = []
        imtafe_list = []
        # loop through every representation layer
        for i in range(vl_reps_1.shape[1]):   
            # just want to use the 0th rep (i.e. directly from the transformer) for now
            if i == 1:
                vl0_test = time.time()
                out_dat_vl, out_lbs_vl, losses_vl = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, "adam", linear_learning_rate, vl_reps_1[:,i,:], vl_lab_1, vl_reps_2[:,i,:], vl_lab_2 )
                auc, imtafe = get_perf_stats( out_lbs_vl, out_dat_vl )
                auc_list.append( auc )
                imtafe_list.append( imtafe )
                vl1_test = time.time()
                print( "LCT layer " + str(i) + "- time taken: " + str( np.round( vl1_test - vl0_test, 2 ) ), flush=True, file=logfile )
                print( "auc: " + str( np.round( auc, 4 ) ) + ", imtafe: " + str( round( imtafe, 1 ) ), flush=True, file=logfile )
                np.save( expt_dir + "lct_ep" +str(epoch) + "_r" +str(i) + "_losses.npy", losses_vl )
        auc_epochs.append( auc_list )
        imtafe_epochs.append( imtafe_list )
        print( "---- --- ----" , flush=True, file=logfile )

        # saving the model
        if epoch % 10 == 0:
            print("saving out jetCLR model", flush=True, file=logfile)
            tms0 = time.time()
            torch.save(net.state_dict(), expt_dir + "model_ep" + str(epoch) + ".pt")
            tms1 = time.time()
            print( f"time taken to save model: {round( tms1-tms0, 1 )}s", flush=True, file=logfile )
        
        # saving out training stats
        if epoch % 10 == 0:
            print( "saving out data/results", flush=True, file=logfile )
            tds0 = time.time()
            np.save( expt_dir + "clr_losses.npy", losses )
            np.save( expt_dir + "auc_epochs.npy", np.array( auc_epochs ) )
            np.save( expt_dir + "imtafe_epochs.npy", np.array( imtafe_epochs ) )
            np.save( expt_dir + "align_loss_train.npy", loss_align_epochs )
            np.save( expt_dir + "uniform_loss_train.npy", loss_uniform_epochs )
            tds1 = time.time()
            print( f"time taken to save data: {round( tds1-tds0, 1 )}s", flush=True, file=logfile )

t2 = time.time()

print( "JETCLR TRAINING DONE, time taken: " + str( np.round( t2-t1, 2 ) ), flush=True, file=logfile )

# save out results
print( "saving out data/results", flush=True, file=logfile )
np.save( expt_dir+"clr_losses.npy", losses )
np.save( expt_dir+"auc_epochs.npy", np.array( auc_epochs ) )
np.save( expt_dir+"imtafe_epochs.npy", np.array( imtafe_epochs ) )
np.save( expt_dir+"align_loss_train.npy", loss_align_epochs )
np.save( expt_dir+"uniform_loss_train.npy", loss_uniform_epochs )

# save out final trained model
print( "saving out final jetCLR model", flush=True, file=logfile )
torch.save(net.state_dict(), expt_dir+"final_model.pt")

print( "starting the final LCT run", flush=True, file=logfile )

# evaluate the network on the testing data, applying some augmentations first if it's required
if args.trs:
    vl_dat_1 = translate_jets( vl_dat_1, width=args.trsw )
    vl_dat_2 = translate_jets( vl_dat_2, width=args.trsw )
with torch.no_grad():
    net.eval()
    #vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
    #vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
    vl_reps_1 = net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
    vl_reps_2 = net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
    net.train()

# final LCT for each rep layer
for i in range(vl_reps_1.shape[1]):
    t3 = time.time()
    out_dat_f, out_lbs_f, losses_f = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, vl_reps_1[:,i,:], vl_lab_1, vl_reps_2[:,i,:], vl_lab_2 )
    auc, imtafe = get_perf_stats( out_lbs_f, out_dat_f )
    ep=0
    step_size = 25
    for lss in losses_f[::step_size]:
        print( f"(rep layer {i}) epoch: " + str( ep ) + ", loss: " + str( lss ), flush=True, file=logfile )
        ep+=step_size
    print( f"(rep layer {i}) auc: "+str( round(auc, 4) ), flush=True, file=logfile )
    print( f"(rep layer {i}) imtafe: "+str( round(imtafe, 1) ), flush=True, file=logfile )
    t4 = time.time()
    np.save( expt_dir+f"linear_losses_{i}.npy", losses_f )
    np.save( expt_dir+f"test_linear_cl_{i}.npy", out_dat_f )

print( "final LCT  done and output saved, time taken: " + str( np.round( t4-t3, 2 ) ), flush=True, file=logfile )
print("............................", flush=True, file=logfile)

t5 = time.time()

print( "----------------------------", flush=True, file=logfile )
print( "----------------------------", flush=True, file=logfile )
print( "----------------------------", flush=True, file=logfile )
print( "ALL DONE, total time taken: " + str( np.round( t5-t0, 2 ) ), flush=True, file=logfile )
