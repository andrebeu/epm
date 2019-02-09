## NBACK_CLASSIFICATION BRANCH 
import sys,os
from glob import glob as glob
import numpy as np
from nback import *



nback = int(sys.argv[1])
stsize = 50
depth = 50

NUM_EPOCHS = 100000
EPOCHS_PER_SESSION = 500

## initialize
ML = MetaLearner(stsize,depth,nback)
trainer = Trainer(ML)

# train
train_loss,train_acc = trainer.train_loop(NUM_EPOCHS,EPOCHS_PER_SESSION)

# save
model_name = 'state_%i-depth_%i-nback_%i'%(stsize,depth,nback)
num_models = len(glob('models/sweep_N/%s/*'%model_name)) 
model_dir = 'models/sweep_N/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

ML.saver_op.save(ML.sess,model_dir+'/final')
np.save(model_dir+"/train_loss-"+model_name,train_loss)
np.save(model_dir+"/train_acc-"+model_name,train_acc)

