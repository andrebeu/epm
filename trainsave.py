import sys
import numpy as np
import tensorflow as tf

from dualPM import *

# param inputs
stsize = int(sys.argv[1])
nback = int(sys.argv[2])
num_og_tokens = int(sys.argv[3])
num_trials_pm = int(sys.argv[4])
seed = int(sys.argv[5])

model_name = "lstm_%i-nback_%i-ogtokens_%i-pmtrials_%i-seed_%i"%(
                  stsize,nback,num_og_tokens,num_trials_pm,seed)
model_fpath = "model_data/"+model_name
print('--',model_name)

# initialize
net = PMNet(stsize,num_og_tokens,seed)
task = NBackPMTask(nback,num_og_tokens,num_trials_pm)
trainer = Trainer(net,task)

# train
nepochs = 1000
ntrials_per_epoch = 20
train_acc,train_cum_rands = trainer.train_closed_loop(nepochs,ntrials_per_epoch,thresh=.99)

#eval
eval_acc = trainer.eval_loop(500,ntrials=20)

# save
np.save(model_fpath+'-train_acc',train_acc)
np.save(model_fpath+'-train_cum_rands',train_cum_rands)
np.save(model_fpath+'-eval_acc',eval_acc)
net.saver_op.save(net.sess,model_fpath)
