import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, RNN
from tensorflow.keras import Model

class AtBatCell(tf.keras.layers.Layer):
    
    def __init__(self, n_batters, n_pitchers, situation_size, states):
        super(AtBatCell, self).__init__()
        self.n_batters = n_batters
        self.n_pitchers = n_pitchers
        self.situation_size = situation_size
        self.states = states
        self.state_size = tf.TensorShape([self.n_batters + self.n_pitchers, states])

    def build( self, input_shape ):
        self.Wz = self.add_weight('input z',
                                    shape=[self.states*2, self.situation_size])
        self.Wr = self.add_weight('input r',
                                    shape=[self.states*2, self.situation_size])
        self.Wh = self.add_weight('input h',
                                    shape=[self.states*2, self.situation_size])
        self.Uz = self.add_weight('state z',
                                    shape=[self.states*2, self.states*2])
        self.Ur = self.add_weight('state r',
                                    shape=[self.states*2, self.states*2])
        self.Uh = self.add_weight('state h',
                                    shape=[self.states*2, self.states*2])
        self.bz = self.add_weight('const z',
                                    shape=[self.states*2, 1])
        self.br = self.add_weight('const r',
                                    shape=[self.states*2, 1])
        self.bh = self.add_weight('const h',
                                    shape=[self.states*2, 1])
        super(AtBatCell, self).build(input_shape)
    
    def call(self, inputs, state):
        x, b, p = inputs
        x = tf.reshape(x, (-1, 1))
        state = state[0] + tf.zeros((state[0].shape))
        h = self.get_states(state, b, p)
        hp = self.GRU(x, h)
        state = self.update_states(state, b, p, hp, h)
        state = tf.reshape(state, (1, state.shape[0], state.shape[1]))
        return state, [state]

    def GRU(self, x, h):
        z = self.Wz @ x + self.Uz @ h + self.bz
        z = tf.math.sigmoid(z)
        r = self.Wr @ x + self.Ur @ h + self.br
        r = tf.math.sigmoid(r)
        m = self.Wh @ x + self.Uh @ (r * h) + self.bh
        m = tf.math.tanh(m)
        hp = z * h + (1-z) * m
        return hp

    def get_states(self, states, batter, pitcher):
        state_batter = tf.gather_nd(states[0], batter)
        state_pitcher = tf.gather_nd(states[0], pitcher)
        h = tf.concat((state_batter, state_pitcher), axis=0)
        return tf.reshape(h, (-1, 1))
    
    def update_states(self, states, batter, pitcher, hp, h):
        dh = hp - h
        dh = tf.reshape(dh, (2, -1))
        indices = tf.reshape([batter, pitcher], (2, 1))
        states = states[0] + tf.scatter_nd(indices, dh, states[0].shape)
        return states

class AtBatRNN(RNN):
    
    def __init__(self, n_batters, n_pitchers, situation_size, states, 
                 return_sequences=True, return_state=False,
                 stateful=True, unroll=False):
        self.n_batters = n_batters
        self.n_pitchers = n_pitchers
        self.situation_size = situation_size
        self.states = states
        cell = AtBatCell(n_batters, n_pitchers, situation_size, states)
        super(AtBatRNN, self).__init__(cell, 
            return_sequences=return_sequences, return_state=return_state,
            stateful=stateful, unroll=unroll)
        
    def call(self, inputs, initial_state=None, constants=None):
        return super(AtBatRNN, self).call(inputs, initial_state=initial_state, constants=constants)

class PredictionLayer(tf.keras.layers.Layer):
    
    def __init__(self, n_batters, n_pitchers, situation_size, states, classes):
        super(PredictionLayer, self).__init__()
        self.n_batters = n_batters
        self.n_pitchers = n_pitchers
        self.situation_size = situation_size
        self.states = states
        self.classes = classes

    def build( self, input_shape ):
        self.W = self.add_weight('input',
                                   shape=[self.classes, self.situation_size])
        self.U = self.add_weight('state',
                                   shape=[self.classes, self.states*2])
        self.b = self.add_weight('const',
                                   shape=[self.classes, 1])
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs):
        xp, states, bp, pp = inputs
        hp = self.get_states(states, bp, pp)
        return self.W @ xp + self.U @ hp + self.b

    def get_states(self, states, batters, pitchers):
        seq_len = batters.shape[1]
        state_batters = tf.gather_nd(states[0], batters)
        state_pitchers = tf.gather_nd(states[0], pitchers)
        h = tf.concat((state_batters, state_pitchers), axis=1)
        return tf.reshape(h, (seq_len, -1, 1))

def build_model( n_batters, n_pitchers, situation_size, states, output_shape, seq_len ):
    
    '''
    This function builds the tensorflow model.
    '''
    b = Input(batch_shape=(1, seq_len, 1), dtype=tf.int32)
    p = Input(batch_shape=(1, seq_len, 1), dtype=tf.int32)
    x = Input(batch_shape=(1, seq_len, situation_size))
    
    h_p = AtBatRNN(n_batters, n_pitchers, situation_size, states)(tuple([x, b, p]))
    
    b_p = Input(batch_shape=(None, seq_len, 1), dtype=tf.int32)
    p_p = Input(batch_shape=(None, seq_len, 1), dtype=tf.int32)
    x_p = Input(batch_shape=(None, seq_len, situation_size, 1))
    
    y_p = PredictionLayer(n_batters, n_pitchers, situation_size, states, output_shape)([x_p, h_p, b_p, p_p])
    
    y_p = Activation('softmax')(y_p)

    model = Model(inputs=[b, p, x, b_p, p_p, x_p],
                  outputs=[y_p])
    
    return model