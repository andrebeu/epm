import tensorflow as tf
import numpy as np

BATCH_SIZE = 1


class NMBackTask():

  def __init__(self,nback,mback,nstim):
    """ assume m>n
    """
    self.nmbackL = [nback,mback]
    self.nstim = nstim
    return None

  def gen_seq(self,ntrials,task_flag):
    """
    task_flag can be 0 or 1 to indicate whether
      to perform nback or mback.
    returns X=[[x(t),y(t-t)],...] Y=[[[y(t)]]]
        `batch,time,step`
    """
    # sample either n or mback task
    nmback = self.nmbackL[task_flag]
    # compose sequnece
    seq = np.random.randint(2,self.nstim+2,ntrials)
    seq_roll = np.roll(seq,nmback)
    Xt = seq
    Yt = (seq==seq_roll).astype(int)
    Xt = np.append(task_flag,Xt)
    Yt = np.append(task_flag,Yt)
    X = np.expand_dims(Xt[:-1],0) 
    Y = np.expand_dims(Yt[:-1],0)
    return X,Y

class NBackPMTask():

  def __init__(self,nback,nstim):
    """ assume m>n
    """
    self.nback = nback
    self.nstim = nstim
    return None

  def gen_seq(self,ntrials):
    """
    task_flag can be 0 or 1 to indicate whether
      to perform nback or mback.
    returns X=[[x(t),y(t-t)],...] Y=[[[y(t)]]]
        `batch,time,step`
    """
    # compose sequnece
    seq = np.random.randint(1,self.nstim+1,ntrials)
    seq_roll = np.roll(seq,self.nback)
    Xt = seq
    Yt = (seq==seq_roll).astype(int)
    # Xt = np.append(task_flag,Xt)
    # Yt = np.append(task_flag,Yt)
    X = np.expand_dims(Xt[:-1],0) 
    Y = np.expand_dims(Yt[:-1],0)
    return X,Y


class PureEM():

  def __init__(self,ntrials,nstim,dim):
    self.ntrials = ntrials
    self.nstim = nstim
    self.dim = dim
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.build()
    return None

  def build(self):
    with self.graph.as_default():
      ## model inputs
      self.trial_ph,self.stim_ph,self.y_ph = self.setup_placeholders()
      self.trial_embed,self.stim_embed = self.get_input_embeds()
      ## compute internal state
      internal_state = tf.keras.layers.Dense(self.dim,activation='sigmoid')(self.stim_embed)
      ## Episodic memory [internal_state,trial_embed]
      self.M_keys,self.M_values = self.get_memory_mats()
      # retrieve
      self.retrieved_memory = retrieved_memory = self.retrieve_memory(internal_state)
      # write {internal_state:trial_embed}
      write_to_memory_value = tf.assign(
                                self.M_values[tf.squeeze(self.trial_ph),:],
                                tf.squeeze(internal_state))
      write_to_memory_keys = tf.assign(
                                self.M_keys[tf.squeeze(self.trial_ph),:],
                                tf.squeeze(self.trial_embed))
      self.write_to_memory = tf.group([write_to_memory_value,write_to_memory_keys])
      # empty memory
      zero_K = self.M_keys.initializer
      zero_V = self.M_values.initializer
      self.empty_memory = tf.group([zero_K,zero_V])
      ## compute response
      response_in = tf.concat([internal_state,retrieved_memory],axis=-1)
      response_in = tf.keras.layers.Dense(self.dim,activation='sigmoid')(response_in)
      response_logits = tf.keras.layers.Dense(2,activation=None)(response_in)
      ## loss and optimization
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=tf.one_hot(self.y_ph,2),
                          logits=response_logits)
      self.minimizer = tf.train.AdamOptimizer(0.05).minimize(self.train_loss)
      ## eval
      self.response_sm = tf.nn.softmax(response_logits)
      self.response = tf.argmax(self.response_sm,axis=1)
      ## extra
      self.sess.run(tf.global_variables_initializer())
    return None

  def reinitialize(self):
    """ reinitializes variables to reset weights"""
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
    return None

  def retrieve_memory(self,query_key):
    ## use internal state to retrieve memory
    # compare internal state to stored internal states (keys)
    state_Mkeys_sim = tf.keras.metrics.cosine(query_key,self.M_keys)
    state_Mkeys_sim = tf.expand_dims(state_Mkeys_sim,axis=0)
    state_Mkeys_sim = tf.nn.softmax(state_Mkeys_sim)
    self.state_Mkeys_sim = state_Mkeys_sim
    # use similarity to form memory retrieval
    retrieved_memory = tf.transpose(
                        tf.matmul(
                          self.M_values,state_Mkeys_sim,
                          transpose_a=True,transpose_b=True
                        ))
    return retrieved_memory

  def setup_placeholders(self):
    trial_ph = tf.placeholder(name='trial_ph',shape=[1],dtype=tf.int32)
    stim_ph = tf.placeholder(name='stim_ph',shape=[1],dtype=tf.int32)
    y_ph = tf.placeholder(name='true_y_ph',shape=[1],dtype=tf.int32)
    return trial_ph,stim_ph,y_ph

  def get_input_embeds(self):
    # setupmat
    self.trial_emat = trial_emat = tf.get_variable('trial_emat',[self.ntrials,self.dim],
                    trainable=True,initializer=tf.initializers.identity()) 
    stim_emat = tf.get_variable('stimulus_emat',[self.nstim,self.dim],
                    trainable=True,initializer=tf.initializers.identity())
    # lookup
    trial_embed = tf.nn.embedding_lookup(trial_emat,self.trial_ph,name='trial_embed')
    stim_embed = tf.nn.embedding_lookup(stim_emat,self.stim_ph,name='stim_embed')
    return trial_embed,stim_embed


  def get_memory_mats(self):
    """ {internal_state: trial_embed}
    """
    M_keys = tf.get_variable('memory_matrix_keys',[self.ntrials,self.dim],
                trainable=False,initializer=tf.initializers.zeros())
    M_values = tf.get_variable('memory_matrix_values',[self.ntrials,self.dim],
                trainable=False,initializer=tf.initializers.zeros())
    return M_keys,M_values