from glob import glob as glob
import numpy as np
from nback import *

from datetime import datetime as dt
import matplotlib.pyplot as plt


for nback in range(2,5):
	plt.close('all')
	plt.figure(figsize=(14,4))
	for idx in range(1,11):
	  stsize=idx*10
	  print(stsize)
	  ML = MetaLearner(stsize=stsize,depth=40,nback=nback)
	  trainer=Trainer(ML)
	  trainer.train_loop(2000,200)
	  eval_loss,eval_acc = trainer.eval_loop(500)
	  plt.plot(eval_acc.mean(0),c=plt.get_cmap('Blues')((idx+3)*25),label=stsize)
	  plt.set_ylim(.3,.9)

	plt.title('sweep_stsize %iback with 3 tokens - 10 train sessions, 200 epochs each, embed_size=8'%(nback))
	plt.legend()
	plt.savefig('figures/sweep_stsize_%iback'%nback)