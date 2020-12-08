<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras import backend as K


class WeightedLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
      super(WeightedLayer, self).__init__(**kwargs)

      self.w1 = self.add_weight(name='w1', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)
      self.w2 = self.add_weight(name='w2', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)                  

    def call(self, inputs1, inputs2):
      return inputs1 * self.w1 + inputs2 * self.w2

    def get_config(self):
      config = super(WeightedLayer, self).get_config()
=======
import tensorflow as tf
from tensorflow.keras import backend as K


class WeightedLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
      super(WeightedLayer, self).__init__(**kwargs)

      self.w1 = self.add_weight(name='w1', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)
      self.w2 = self.add_weight(name='w2', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)                  

    def call(self, inputs1, inputs2):
      return inputs1 * self.w1 + inputs2 * self.w2

    def get_config(self):
      config = super(WeightedLayer, self).get_config()
>>>>>>> 65f55c70c0449874e6f1d72320a242641fb2d0f4
      return config