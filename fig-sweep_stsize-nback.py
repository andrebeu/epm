from glob import glob as glob
import numpy as np
from nback import *

from datetime import datetime as dt
import matplotlib.pyplot as plt

stsizeL = [1,5,10,15,20,25]
num_groups = len(stsizeL)
for nback in range(2,6):
  fig,axarr = plt.subplots(2,1,figsize=(14,8));axarr=axarr.reshape(-1)
  idx = 0
  for stsize in stsizeL:
    idx += 1
    print(nback,stsize)
    ML = MetaLearner(stsize=stsize,depth=30,nback=nback)
    trainer=Trainer(ML)
    trainer.train_loop(4000,200)
    eval_loss,eval_acc = trainer.eval_loop(500)
    axarr[0].plot(eval_acc.mean(0),c=plt.get_cmap('Blues')((idx+3)*25),label=stsize)
    eval_acc = eval_acc[:,10:]
    axarr[1].bar(idx,eval_acc.mean(),yerr=eval_acc.mean(0).std(),color=plt.get_cmap('Blues')((idx+3)*25))
  axarr[1].set_xticks(1+np.arange(num_groups))
  axarr[1].set_xticklabels(stsizeL)
  axarr[0].set_ylim(.3,.9)
  axarr[1].set_ylim(0,1)
  axarr[0].axhline(0.66,c='red',lw=.4)
  axarr[0].axhline(0.33,c='red',lw=.4)
  axarr[0].set_title('sweep_stsize %iback with 3 tokens - 20 train sessions, 200 epochs each, embed_size=8'%(nback))
  axarr[0].legend()
  plt.savefig('figures/bar+line-sweep_stsize-%iback'%nback)
  plt.close('all')