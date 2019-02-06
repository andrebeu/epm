from nback import *
import os,sys
from glob import glob as glob



STATE_SIZE = sys.argv[1]
DEPTH = 30
NUM_STIM = 30
NUM_BACK = 4
EPOCHS_PER_EPISODE = 500
NUM_EPOCHS = 50000


ML = MetaLearner(STATE_SIZE,DEPTH,NUM_STIM)
trainer = Trainer(ML,NUM_BACK)
trainer.train_loop(NUM_EPOCHS,EPOCHS_PER_EPISODE)

# eval
loss_arr = trainer.eval_loop(4000)


# saving

model_name='state_%i-depth_%i-epochs_%i-epochsperepisode_%i'%(
  STATE_SIZE,DEPTH,NUM_EPOCHS,EPOCHS_PER_EPISODE)

num_models = len(glob('models/%s/*'%model_name))
fpath = 'models/%s/%i'%(model_name,num_models+1) 
os.makedirs(fpath)

ML.saver_op.save(ML.sess,fpath+'/final_model')
np.save(fpath+"/"+model_name,lossL)