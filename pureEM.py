import tensorflow as tf
import numpy as np

TRAIN_PMEMBED = False

class NBackPMTask():

  def __init__(self,nback,num_og_tokens,num_trials_pm,seed=0):
    """ 
    """
    np.random.seed(seed)
    self.nback = nback
    self.num_og_tokens = num_og_tokens
    self.pm_token = num_og_tokens
    self.min_start_trials = 1
    self.num_trials_pm = num_trials_pm
    return None

  def gen_seq(self,ntrials=30,pm_trial_position=None):
    """
    if pm_trial_position is not specified, they are randomly sampled
      rand pm_trial_position for training, fixed for eval
    """
    # insert ranomly positioned pm trials
    if type(pm_trial_position)==type(None):
      ntrials -= 1+self.num_trials_pm
      pm_trial_position = np.random.randint(self.min_start_trials,ntrials,self.num_trials_pm) 
    else:
      ntrials -= 1+len(pm_trial_position)
      pm_trial_position = pm_trial_position
    # generate og stim
    seq = np.random.randint(0,self.num_og_tokens,ntrials)
    X = np.insert(seq,[0,*pm_trial_position],self.pm_token)
    # form Y 
    Xroll = np.roll(X,self.nback)
    Y = (X == Xroll).astype(int) # nback trials
    Y[X==self.pm_token]=2 # pm trials
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
    self.num_og_tokens = num_og_tokens
    self.nstim = num_og_tokens+1 
    self.num_actions = 3
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.seed)
      ## data feeding
      self.setup_placeholders()
      self.xbatch_id,self.ybatch_id,self.itr_initop = self.data_pipeline() 
      # embedding matrix and randomization op
      
      if TRAIN_PMEMBED:
        print('trainable pm embed')
        self.stim_emat = tf.get_variable(name='og_emat',shape=[self.num_og_tokens,self.edim],
                      trainable=False,initializer=tf.initializers.random_uniform()) 
        self.pm_emat = tf.get_variable(name='pm_emat',shape=[1,self.edim],
                      trainable=True,initializer=tf.initializers.random_uniform()) 
        self.randomize_emat = self.stim_emat.initializer
        self.emat = tf.concat([self.stim_emat,self.pm_emat],axis=0)
      else:
        print('randomizing pm embed')
        self.emat = tf.get_variable(name='og_emat',shape=[self.nstim,self.edim],
                      trainable=False,initializer=tf.initializers.random_uniform()) 
        self.randomize_emat = self.emat.initializer
      ## inference
      self.xbatch = tf.nn.embedding_lookup(self.emat,self.xbatch_id,name='xembed') 
      self.y_logits = self.PureWM(self.xbatch) 
      self.ybatch_onehot = tf.one_hot(self.ybatch_id,self.num_actions) 
      ## train
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,
                          logits=self.y_logits)
      self.minimizer = tf.train.AdamOptimizer(0.0005).minimize(self.train_loss)
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

  def PureWM(self,xbatch):
    """
    """
    print('lstm2, in+out layers with do_')
    # input projection with dropout
    xbatch = tf.keras.layers.Dense(self.stsize,activation='relu')(xbatch)
    xbatch = tf.keras.layers.Dropout(self.dropout_rate)(xbatch)
    # trinable initial states
    init_cstate1 = tf.get_variable(name='init_cstate1',
                  shape=[1,self.stsize],trainable=True)
    init_hstate1 = tf.get_variable(name='init_hstate1',
                  shape=[1,self.stsize],trainable=True)
    init_cstate2 = tf.get_variable(name='init_cstate2',
                  shape=[1,self.stsize],trainable=True)
    init_hstate2 = tf.get_variable(name='init_hstate2',
                  shape=[1,self.stsize],trainable=True)
    init_state1 = [init_cstate1,init_hstate1]
    init_state2 = [init_cstate2,init_hstate2]
    ## lstm layers
    lstm_layer1 = tf.keras.layers.LSTM(self.stsize,return_sequences=True)
    lstm_outputs = lstm_layer1(xbatch,initial_state=init_state1)
    # lstm_layer2 = tf.keras.layers.LSTM(self.stsize,return_sequences=True)
    # lstm_outputs = lstm_layer2(lstm_outputs,initial_state=init_state2)
    # lstm_cell1 = tf.keras.layers.LSTMCell(self.stsize)
    # lstm_cell2 = tf.keras.layers.LSTMCell(self.stsize)
    # lstm_layer = tf.keras.layers.RNN([lstm_cell1,lstm_cell2],return_sequences=True)
    # init_state = lstm_layer.get_initial_state(inputs=xbatch)
    # init_state = [init_cstate,init_hstate]
    # lstm_outputs = lstm_layer(xbatch)
    ## readout layers
    lstm_outputs = tf.keras.layers.Dense(self.stsize,activation='relu')(lstm_outputs)
    lstm_outputs = tf.keras.layers.Dropout(self.dropout_rate)(lstm_outputs)
    y_logits = tf.keras.layers.Dense(self.num_actions,activation=None)(lstm_outputs)
    return y_logits

  def WM_EMthresholded(self,xbatch):
    """
    """
    print(xbatch)
    # input projection with dropout
    xbatch = tf.keras.layers.Dense(self.stsize,activation='relu')(xbatch)
    xbatch = tf.keras.layers.Dropout(self.dropout_rate)(xbatch)
    # trinable initial states
    init_cstate = tf.get_variable(name='init_cstate',
                  shape=[1,self.stsize],trainable=True)
    init_hstate = tf.get_variable(name='init_hstate',
                  shape=[1,self.stsize],trainable=True)
    init_state = [init_cstate,init_hstate]
    # lstm cell
    lstm_layer = tf.keras.layers.LSTM(self.stsize,return_sequences=True)
    lstm_outputs = lstm_layer(xbatch,initial_state=init_state)
    ## readout layers
    lstm_outputs = tf.keras.layers.Dense(self.stsize,activation='relu')(lstm_outputs)
    lstm_outputs = tf.keras.layers.Dropout(self.dropout_rate)(lstm_outputs)
    y_logits = tf.keras.layers.Dense(self.num_actions,activation=None)(lstm_outputs)
    return y_logits


class Trainer():

  def __init__(self,net,task):
    self.net = net
    self.task = task
    return None

  def train_step(self,Xdata,Ydata,cell_state=None):
    feed_dict = { self.net.xph:Xdata,
                  self.net.yph:Ydata,
                  self.net.dropout_rate:0.1,
                  }
    # initialize iterator
    self.net.sess.run([self.net.itr_initop],feed_dict)
    # update weights and compute final loss
    self.net.sess.run(self.net.minimizer,feed_dict)
    return None

  def train_closed_loop(self,nepochs,ntrials_per_epoch=30,thresh=.9):
    train_acc = -np.ones([nepochs])
    train_cum_rands = -np.ones([nepochs])
    nrands = 0
    for ep_num in range(nepochs):
      # train step
      Xdata,Ydata = self.task.gen_seq(ntrials_per_epoch)
      self.train_step(Xdata,Ydata)
      step_acc = self.eval_step(Xdata,Ydata).mean()
      train_acc[ep_num] = step_acc
      train_cum_rands[ep_num] = nrands
      # maybe randomize
      if step_acc >= thresh:
        nrands += 1
        self.net.sess.run(self.net.randomize_emat)
      # printing
      if ep_num%(nepochs/20) == 0:
        print(ep_num/nepochs,train_acc[ep_num].round(2),nrands) 
    return train_acc,train_cum_rands

  def train_loop(self,nepochs,eps):
    """
    """
    train_acc = np.empty([nepochs])
    for ep_num in range(nepochs):
      if ep_num%eps == 0:
        self.net.sess.run(self.net.randomize_emat)
      # train step
      Xdata,Ydata = self.task.gen_seq(NTRIALS_TRAIN)
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
                  self.net.dropout_rate:0.0,
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

  def eval_loop(self,nepisodes,ntrials=20):
    """
    preset pm trial positions for evaluating
    """
    acc_arr = np.zeros([nepisodes,ntrials])
    pm_trial_position = np.array([10-1,15-2]) # eval pm trial position
    for it in range(nepisodes):
      self.net.sess.run(self.net.randomize_emat)
      Xdata_eval,Ydata_eval = self.task.gen_seq(ntrials,pm_trial_position)
      step_acc = self.eval_step(Xdata_eval,Ydata_eval)
      acc_arr[it] = step_acc
    return acc_arr

