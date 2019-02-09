## NBACK_CLASSIFICATION BRANCH 
import sys
import numpy as np
from nback import *


nback = int(sys.argv[1])
stsize = 50
depth = 50

NUM_EPOCHS = 20000
EPOCHS_PER_SESSION = 500

## initialize
ML = MetaLearner(stsize,depth,nback)
trainer = Trainer(ML,nback)

# train
train_loss,train_acc = trainer.train_loop(NUM_EPOCHS,EPOCHS_PER_SESSION)

# save
model_dir = 'models/sweep_N/state_%i-depth_%i-nback_%i/'%(stsize,depth,nback)
ML.saver_op.save(ML.sess,model_dir+'final')
np.save(model_dir+'train_loss',train_loss)
np.save(model_dir+'train_acc',train_acc)
