import sys
import numpy as np
import tensorflow as tf

from pureEM import *

# param inputs
stsize = int(sys.argv[1])
nback = int(sys.argv[2])
num_og_tokens = int(sys.argv[3])
num_pm_trials = int(sys.argv[4])
seed = int(sys.argv[5])

# initialize
net = PMNet(stsize,num_og_tokens,seed)
task = NBackPMTask(nback,num_og_tokens,num_pm_trials,seed)
trainer = Trainer(net,task)

### train params
num_train_sessions = 20
train_epochs = 500
eval_episodes = 500
eval_trials_per_episode = 25
# train/eval/save loop
for tsess in range(1,num_train_sessions+1):
  # train
  trainer.train_loop(train_epochs,train_epochs)
  # eval
  eval_acc = trainer.eval_loop(eval_episodes,eval_trials_per_episode)
  # save
  model_name = "lstm_%i-nback_%i-ogtokens_%i-pmtrials_%i-seed_%i-trepochs_%i-eval_acc"%(
                  stsize,nback,num_og_tokens,num_pm_trials,seed,tsess*train_epochs)
  model_fpath = "model_data/nback+pm/"+model_name
  np.save(model_fpath,eval_acc)
  net.saver_op.save(net.sess,model_fpath)

