## NBACK_CLASSIFICATION BRANCH 

import numpy as np
from nback import *

cell_size = 30
depth = 20
num_stim = 2
numback = 10

NUM_EPOCHS = 2000
EPOCHS_PER_SESSION = 2000 

# initialize
ML = MetaLearner(cell_size,depth=depth,num_stim=num_stim,preunroll=numback)
trainer = Trainer(ML,numback)

# train
train_loss,train_acc = trainer.train_loop(NUM_EPOCHS,EPOCHS_PER_SESSION)

# save
model_dir = 'models/sweep_N/state_%i-depth_%i-numstim_%i-nback_%i/'%(cell_size,depth,num_stim,numback)
ML.saver_op.save(ML.sess,model_dir+'final')
np.save(model_dir+'train_loss',train_loss)
np.save(model_dir+'train_acc',train_loss)
