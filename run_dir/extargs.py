#!/bin/env python3
logfile = "/path/to/logfile.txt"
tr_dat_path = "/path/to/data.npy"
tr_lab_path = "/path/to/labels.npy"
nconstit = 50
model_dim = 1000
output_dim = 1000
n_heads = 4
dim_feedforward = 1000
n_layers = 4
n_head_layers = 2
opt = "adam"
sbratio = 1.0
n_epochs = 500
learning_rate = 0.00005
batch_size = 128
temperature = 0.10
rot = True
ptd = True
ptcm = 0.1
ptst = 0.1
trs = True
trsw = 1.0
cf = True
mask= False
cmask = True
expt = "experiment-name"

