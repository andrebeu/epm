import tensorflow as tf
import numpy as np


TRIALS_TOTAL = 40

# TRIALS_PM = 5
# NUM_OG_TOKENS = 2
# NBACK = 2


class NBackPMTask():

  def __init__(self,nback,num_og_tokens,num_pm_trials,seed=0):
    """ assume m>n
    """
    np.random.seed(seed)
    self.nback = nback
    self.num_og_tokens = num_og_tokens
    self.pm_token = num_og_tokens
    self.min_start_trials = 5
    self.num_pm_trials = num_pm_trials
    return None

  def gen_seq(self,ntrials=TRIALS_TOTAL,pm_trial_position=None):
    """
    if pm_trial_position is not specified, they are randomly sampled
      rand pm_trial_position for training, fixed for eval
    """
    # insert ranomly positioned pm trials
    if type(pm_trial_position)==type(None):
      ntrials -= 1+self.num_pm_trials
      pm_trial_position = np.random.randint(self.min_start_trials,ntrials,self.num_pm_trials) 
    else:
      ntrials -= 1+len(pm_trial_position)
      pm_trial_position = pm_trial_position
    # generate og stim
    seq = np.random.randint(0,self.num_og_tokens,ntrials)
    X = np.insert(seq,[0,*pm_trial_position],self.pm_token)
    # form Y 
    Xroll = np.roll(X,self.nback)
    Y = (X == Xroll).astype(int) # nback trials
    Y[X==2]=2 # pm trials
    # include batch dim
    X = np.expand_dims(X,0)
    Y = np.expand_dims(Y,0)
    return X,Y


class PMNet():

  def __init__(self,stsize,num_og_tokens,seed=0):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.seed = seed
    # net params
    self.stsize = stsize
    self.edim = 8
    self.nstim = num_og_tokens+1 
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.seed)
      ## data feeding
      self.setup_placeholders()
      self.xbatch_id,self.ybatch_id,self.itr_initop = self.data_pipeline() 
      # embedding matrix and randomization op
      self.emat = tf.get_variable(name='emat',shape=[self.nstim,self.edim],
                    trainable=False,initializer=tf.initializers.random_normal(0,1)) 
      self.randomize_emat = self.emat.initializer
      ## inference
      self.xbatch = tf.nn.embedding_lookup(self.emat,self.xbatch_id,name='xembed') 
      self.y_logits = self.RNN_keras(self.xbatch) 
      self.ybatch_onehot = tf.one_hot(self.ybatch_id,self.nstim) 
      ## train
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,
                          logits=self.y_logits)
      self.minimizer = tf.train.AdamOptimizer(0.005).minimize(self.train_loss)
      ## eval
      self.yhat_sm = tf.nn.softmax(self.y_logits)       
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
    dataset = dataset.batch(1)
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    itr_initop = iterator.make_initializer(dataset)
    xbatch,ybatch = iterator.get_next() 
    return xbatch,ybatch,itr_initop

  def setup_placeholders(self):
    """
    setup placeholders as instance variables
    """
    self.xph = tf.placeholder(tf.int32,
              shape=[1,None],
              name="xdata_placeholder")
    self.yph = tf.placeholder(tf.int32,
                  shape=[1,None],
                  name="ydata_placeholder")
    self.dropout_rate = tf.placeholder(tf.float32,
                  shape=[],
                  name="dropout_rate")
    return None

  # inference

  def RNN_keras(self,xbatch):
    """
    NB unlike before no input projection
    """
    # input projection with dropout
    xbatch = tf.keras.layers.Dense(self.stsize,activation='relu')(xbatch)
    xbatch = tf.layers.dropout(xbatch,rate=self.dropout_rate)
    # trinable initial states
    init_cstate = tf.get_variable(name='init_cstate',
                  shape=[1,self.stsize],trainable=True)
    init_hstate = tf.get_variable(name='init_hstate',
                  shape=[1,self.stsize],trainable=True)
    # lstm cell
    lstm_layer = tf.keras.layers.LSTM(self.stsize,return_sequences=True)
    lstm_outputs = lstm_layer(xbatch,initial_state=[init_cstate,init_hstate])
    ## readout layers
    y_logits = tf.keras.layers.Dense(
                      self.nstim,                      
                      activation=None
                      )(lstm_outputs)
    # readout dropout
    y_logits = tf.layers.dropout(y_logits,
                    rate=self.dropout_rate,
                    name='readout')
    return y_logits


class Trainer():

  def __init__(self,net,task):
    self.net = net
    self.task = task
    return None

  def train_step(self,Xdata,Ydata,cell_state=None):
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_rate:0.9,
                  }
    # initialize iterator
    self.net.sess.run([self.net.itr_initop],feed_dict)
    # update weights and compute final loss
    self.net.sess.run(self.net.minimizer,feed_dict)
    return None

  def train_loop(self,nepochs,eps):
    """
    """
    train_acc = np.empty([nepochs])
    for ep_num in range(nepochs):
      if ep_num%eps == 0:
        self.net.sess.run(self.net.randomize_emat)
      # train step
      Xdata,Ydata = self.task.gen_seq()
      self.train_step(Xdata,Ydata)
      step_acc = self.eval_step(Xdata,Ydata)
      train_acc[ep_num] = step_acc.mean()
      # printing
      if ep_num%(nepochs/20) == 0:
        print(ep_num/nepochs,train_acc[ep_num]) 
    return train_acc

  def eval_step(self,Xdata,Ydata):
    ## setup
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_rate:1.0,
                  }
    self.net.sess.run([self.net.itr_initop],feed_dict)
    ## eval
    step_yhat_sm,step_ybatch = self.net.sess.run(
                                        [
                                        self.net.yhat_sm,
                                        self.net.ybatch_onehot
                                        ],feed_dict)
    step_acc = step_yhat_sm.argmax(2) == step_ybatch.argmax(2)
    return step_acc

  def eval_loop(self,nepisodes,ntrials=TRIALS_TOTAL):
    """
    preset pm trial positions for evaluating
    """
    acc_arr = np.zeros([nepisodes,ntrials])
    for it in range(nepisodes):
      self.net.sess.run(self.net.randomize_emat)
      pm_trial_position = np.array([10-1,15-2]) # eval pm trial position
      Xdata_eval,Ydata_eval = self.task.gen_seq(ntrials,pm_trial_position)
      step_acc = self.eval_step(Xdata_eval,Ydata_eval)
      acc_arr[it] = step_acc
    return acc_arr