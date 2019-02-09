import tensorflow as tf
from datetime import datetime as dt
import numpy as np

BATCH_SIZE = 1

""" output 0 for trials previous to N
"""

class NBackTask():

  def __init__(self,nback):
    self.nback = nback
    return None

  def gen_seq(self,ntrials,nstim):
    """
    returns X=[[x(t),y(t-t)],...] Y=[[[y(t)]]]
        `batch,time,step`
    """
    # ntrials += self.nback
    seq = np.random.randint(0,nstim,ntrials+self.nback)
    # seq = np.arange(ntrials)
    Xt = seq
    Xroll = np.roll(seq,self.nback)
    Yt = np.array(Xt == Xroll).astype(int)
    X = np.expand_dims(Xt,0) 
    Y = np.expand_dims(Yt,0)
    return X,Y


class MetaLearner():

  def __init__(self,stsize,depth,nback,nstim=2):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.stsize = stsize
    self.depth = depth 
    self.nback = nback
    self.nstim = nstim
    self.embed_size = 2
    self.num_actions = 2
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      ## data feeding
      self.setup_placeholders()
      self.xbatch_id,self.ybatch_id,self.itr_initop = self.data_pipeline() # x(batch,depth+nback), y(batch,depth+nback)
      # embedding matrix and randomization op
      self.emat = tf.get_variable('embedding_matrix',[self.nstim,self.embed_size],
                    trainable=False,initializer=tf.initializers.random_normal(0,1)) 
      self.randomize_emat = self.emat.initializer
      ## inference
      self.xbatch = tf.nn.embedding_lookup(self.emat,self.xbatch_id,name='xembed') # batch,depth+nabck,embsize
      self.yhat_unscaled_full,self.finalstate = self.RNNinference(self.xbatch) # batch,depth+nback,num_actions
      self.yhat_unscaled_bptt = self.yhat_unscaled_full[:,self.nback:,:] # batch,depth,num_actions
      ## train
      self.ybatch_onehot_full = tf.one_hot(self.ybatch_id,self.num_actions) # batch,depth+nback,num_actions
      self.ybatch_onehot_bptt = self.ybatch_onehot_full[:,self.nback:,:] # batch,depth,num_actions
      self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot_bptt,logits=self.yhat_unscaled_bptt)
      print('ADAM005')
      self.minimizer = tf.train.AdamOptimizer(0.005).minimize(self.loss)
      ## eval
      self.yhat_sm = tf.nn.softmax(self.yhat_unscaled_full) # batch,depth+nback,num_actions
      self.eval_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot_full,logits=self.yhat_unscaled_full)
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
              shape=[None,self.depth+self.nback],
              name="xdata_placeholder")
    self.yph = tf.placeholder(tf.int32,
                  shape=[None,self.depth+self.nback],
                  name="ydata_placeholder")
    self.dropout_keep_pr = tf.placeholder(tf.float32,
                  shape=[],
                  name="dropout_ph")
    self.cellstate_ph = tf.placeholder(tf.float32,
                  shape=[None,self.stsize],
                  name = "initialstate_ph")
    return None

  # inference

  def RNNinference(self,xbatch):
    """ 
    """
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # setup RNN cell      
      cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            self.stsize,dropout_keep_prob=self.dropout_keep_pr)
      ## input projection
      # [batch,depth,insteps,embed_size]
      xbatch = tf.layers.dense(xbatch,self.stsize,
                  activation=tf.nn.relu,name='inproj')
      # [batch,depth,insteps,stsize]
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      ## unroll 
      outputL = []
      for tstep in range(self.depth+self.nback):
        if tstep > 0: cellscope.reuse_variables()
        output,state = cell(xbatch[:,tstep,:], state)
        outputL.append(output)
    # output projection
    outputs = tf.convert_to_tensor(outputL) # depth,batch,cellsize
    outputs = tf.transpose(outputs,(1,0,2)) # batch,depth,cellsize
    outputs = tf.layers.dense(outputs,self.stsize,
                  activation=tf.nn.relu,name='hidoutlayer')
    outputs = tf.layers.dense(outputs,self.num_actions,
                  activation=None,name='outproj')
    return outputs,state



class Trainer():

  def __init__(self,net):
    self.task = NBackTask(net.nback)
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
        rand_cell_state = cell_state = np.random.randn(BATCH_SIZE,self.net.stsize)
      # # generate data
      # emat = self.net.sess.run(self.net.emat)
      Xdata,Ydata = self.task.gen_seq(self.net.depth,self.net.nstim)
      # train step
      cell_state = self.train_step(Xdata,Ydata,cell_state)
      # printing
      if ep_num%(num_epochs/num_evals) == 0:
        eval_idx += 1
        evalstep_loss,evalstep_acc = self.eval_step(Xdata,Ydata,cell_state)
        loss_arr[eval_idx] = np.mean(evalstep_loss)
        acc_arr[eval_idx] = np.mean(evalstep_acc)
        if ep_num%(num_epochs/20) == 0:
          print(ep_num/num_epochs,np.mean(evalstep_loss)) 
    return loss_arr,acc_arr

  def eval_step(self,Xdata,Ydata,cell_state=None):
    if type(cell_state) == type(None):
      cell_state = np.random.randn(BATCH_SIZE,self.net.stsize)
    ## setup
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_keep_pr:1.0,
                  self.net.cellstate_ph:cell_state
                  }
    self.net.sess.run([self.net.itr_initop],feed_dict)
    ## eval
    step_loss,step_yhat_sm,step_ybatch = self.net.sess.run(
                                        [self.net.eval_loss,
                                        self.net.yhat_sm,
                                        self.net.ybatch_onehot_full
                                        ],feed_dict)
    step_acc = step_yhat_sm.argmax(2) == step_ybatch.argmax(2)
    return step_loss,step_acc

  def eval_loop(self,num_itr):
    loss_arr = np.empty([num_itr,self.net.depth+self.net.nback])
    acc_arr = np.empty([num_itr,self.net.depth+self.net.nback])
    for it in range(num_itr):
      self.net.sess.run(self.net.randomize_emat)
      rand_cell_state = cell_state = np.random.randn(BATCH_SIZE,self.net.stsize)
      Xdata,Ydata = self.task.gen_seq(self.net.depth,self.net.nstim)
      step_loss,step_acc = self.eval_step(Xdata,Ydata,rand_cell_state)
      loss_arr[it] = step_loss
      acc_arr[it] = step_acc
    return loss_arr,acc_arr
