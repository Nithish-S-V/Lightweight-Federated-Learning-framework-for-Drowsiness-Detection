import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        
    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule,
                   self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W')
            
        self.built = True
        
    def call(self, inputs):
        # inputs.shape = [None, input_num_capsule, input_dim_capsule]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_capsule]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        
        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape = [None, input_num_capsule, num_capsule, 1, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0
        # W.shape = [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape = [num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.scan(lambda x, y: K.batch_dot(y, self.W, [3, 3]),
                           elems=inputs_tiled,
                           initializer=K.zeros([self.input_num_capsule,
                                              self.num_capsule,
                                              1,
                                              self.dim_capsule]))
                                              
        # Routing algorithm
        b = tf.zeros(shape=[K.shape(inputs_hat)[0],
                           self.input_num_capsule,
                           self.num_capsule, 1])
                           
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = squash(K.sum(c * inputs_hat, axis=1, keepdims=True))
            
            if i < self.routings - 1:
                b += K.sum(inputs_hat * outputs, axis=-1, keepdims=True)
                
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_capsule])
        
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def squash(vectors, axis=-1):
    """Squashing function for capsule values"""
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors
