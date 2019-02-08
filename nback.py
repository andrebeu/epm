import tensorflow as tf
from datetime import datetime as dt
import numpy as np


INSTEPS = 1
OUTSTEPS = 1
BATCH_SIZE = 1

""" output 0 for trials previous to N
"""

class NBackTask():

  def __init__(self,numback):
    self.numback=numback
    return None

  def gen_seq(self,num_trials,numstim):
    """
    returns X=[[x(t),y(t-t)],...] Y=[[[y(t)]]]
        `batch,time,step`
    """
    seq = np.random.randint(0,numstim,num_trials)
    # seq = np.arange(num_trials)
    Xt = seq
    Xroll = np.roll(seq,self.numback)
    Yt = np.array(Xt == Xroll).astype(int)
    X = np.expand_dims(np.stack([Xt],1),0) 
    Y = np.expand_dims(Yt,0)
    return X,Y


class MetaLearner():

  def __init__(self,cell_size,depth,numstim,preunroll):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.preunroll_depth = preunroll
    self.depth = depth
    self.cell_size = cell_size
    self.numstim = numstim
    self.embed_size = 2
    self.num_actions = 2
    self.build()
    return None

  def build(self):
    with self.graph.as_default():

      ## data feeding
      self.setup_placeholders()
      self.xbatch_id,self.ybatch_id,self.itr_initop = self.data_pipeline() # x(batches,depth,inlen), y(batch,depth,1)
      self.ybatch_id = self.ybatch_id[:,self.preunroll_depth:]
      # embedding
      self.emat = tf.get_variable('embedding_matrix',[self.numstim,self.embed_size],trainable=False,initializer=tf.initializers.random_normal(0,1)) 
      self.randomize_emat = self.emat.initializer
      ## inference
      self.xbatch = tf.nn.embedding_lookup(self.emat,self.xbatch_id,name='xembed') # batch,bptt,instep,numstim
      self.yhat_unscaled,self.finalstate = self.RNNinference(self.xbatch) # batch,bptt,num_actions
      ## loss
      self.ybatch_onehot = tf.one_hot(self.ybatch_id,self.num_actions) # batch,bptt,num_actions
      self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,logits=self.yhat_unscaled)
      ## eval
      self.yhat_sm = tf.nn.softmax(self.yhat_unscaled)
      self.eval_loss = self.loss
      ##
      print('ADAM005')
      self.minimizer = tf.train.AdamOptimizer(0.005).minimize(self.loss)
      # other
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver(max_to_keep=None)
    return None

  def reinitialize(self):
    """ reinitializes variables to reset weights"""
    with self.graph.as_default():
      print('randomizing params')
      self.sess.run(tf.global_variables_initializer())
    return None
  
  # setup 

  def data_pipeline(self):
    """data pipeline
    returns x,y = get_next
    also creates self.itr_initop
    """
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,self.yph))
    dataset = dataset.batch(BATCH_SIZE)
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    itr_initop = iterator.make_initializer(dataset)
    xbatch,ybatch = iterator.get_next() 
    return xbatch,ybatch,itr_initop

  def setup_placeholders(self):
    """
    setsup placeholders as instance variables
    """
    self.xph = tf.placeholder(tf.int32,
              shape=[None,self.depth,INSTEPS],
              name="xdata_placeholder")
    self.yph = tf.placeholder(tf.int32,
                  shape=[None,self.depth],
                  name="ydata_placeholder")
    self.dropout_keep_pr = tf.placeholder(tf.float32,
                  shape=[],
                  name="dropout_ph")
    self.cellstate_ph = tf.placeholder(tf.float32,
                  shape=[None,self.cell_size],
                  name = "initialstate_ph")
    return None

  # inference

  def RNNinference(self,xbatch):
    """ 
    processes a batch of input sequences
    each sequence is of dim [DEPTH]
    returns ops: output sequence and final state
    output sequence is of length DEPTH
    """
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # setup RNN cell      
      cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            self.cell_size,dropout_keep_prob=self.dropout_keep_pr)
      ## input projection
      # [batch,depth,insteps,embed_size]
      xbatch = tf.layers.dense(xbatch,self.cell_size,
                  activation=tf.nn.relu,name='inproj')
      # [batch,depth,insteps,cell_size]
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      ## unroll 
      outputL = []
      for tstep in range(self.depth):
        for instep in range(INSTEPS):
          if instep > 0: cellscope.reuse_variables()
          output,state = cell(xbatch[:,tstep,instep,:], state)
        if tstep >= self.preunroll_depth:
	        outputL.append(output)
    # output projection
    outputs = tf.convert_to_tensor(outputL) # depth,batch,cellsize
    outputs = tf.transpose(outputs,(1,0,2)) # batch,depth,cellsize
    outputs = tf.layers.dense(outputs,self.cell_size,
                  activation=tf.nn.relu,name='hidoutlayer')
    outputs = tf.layers.dense(outputs,self.num_actions,
                  activation=None,name='outproj')
    return outputs,state



class Trainer():

  def __init__(self,net,numback):
    self.task = NBackTask(numback)
    self.net = net
    return None

  def train_step(self,Xdata,Ydata,cell_state):
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_keep_pr:0.9,
                  self.net.cellstate_ph:cell_state
                  }
    # initialize iterator
    self.net.sess.run([self.net.itr_initop],feed_dict)
    # update weights and compute final loss
    cell_st,_ = self.net.sess.run(
      [self.net.finalstate,self.net.minimizer],feed_dict)
    cell_st = cell_st[0] # only return c-state
    return cell_st

  def train_loop(self,num_epochs,epochs_per_session):
    """
    """
    num_evals = num_epochs
    loss_arr = np.empty([num_evals])
    acc_arr = np.empty([num_evals])
    eval_idx = -1
    for ep_num in range(num_epochs):
      if ep_num%epochs_per_session == 0:
        # randomize embeddings
        self.net.sess.run(self.net.randomize_emat)
        # flush cell state
        rand_cell_state = cell_state = np.random.randn(BATCH_SIZE,self.net.cell_size)
      # # generate data
      # emat = self.net.sess.run(self.net.emat)
      Xdata,Ydata = self.task.gen_seq(self.net.depth,self.net.numstim)
      # train step
      cell_state = self.train_step(Xdata,Ydata,cell_state)
      # printing
      if ep_num%(num_epochs/num_evals) == 0:
        eval_idx += 1
        step_loss,step_yhat,step_ybatch = self.eval_step(Xdata,Ydata,cell_state)
        step_acc = np.mean(step_ybatch.argmax(2) == step_yhat.argmax(2))
        loss_arr[eval_idx] = np.mean(step_loss)
        acc_arr[eval_idx] = step_acc
        if ep_num%(num_epochs/20) == 0:
          print(ep_num/num_epochs,np.mean(step_loss)) 
    return loss_arr,acc_arr

  def eval_step(self,Xdata,Ydata,cell_state=None):
    if type(cell_state) == type(None):
      cell_state = np.random.randn(BATCH_SIZE,self.net.cell_size)
    ## setup
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_keep_pr:1.0,
                  self.net.cellstate_ph:cell_state
                  }
    self.net.sess.run([self.net.itr_initop],feed_dict)
    ## eval
    eval_step_outs = self.net.sess.run([self.net.eval_loss,self.net.yhat_sm,self.net.ybatch_onehot],feed_dict)
    return eval_step_outs

  def eval_loop(self,num_itr):
    loss_arr = np.empty([num_itr,self.net.depth-self.net.preunroll_depth])
    acc_arr = np.empty([num_itr])
    for it in range(num_itr):
      self.net.sess.run(self.net.randomize_emat)
      rand_cell_state = cell_state = np.random.randn(BATCH_SIZE,self.net.cell_size)
      Xdata,Ydata = self.task.gen_seq(self.net.depth,self.net.numstim)
      step_loss,step_yhat,step_ybatch = self.eval_step(Xdata,Ydata,rand_cell_state)
      step_acc = np.sum(step_ybatch.argmax(2) == step_yhat.argmax(2))
      loss_arr[it] = step_loss
      acc_arr[it] = step_acc
    return loss_arr,acc_arr
