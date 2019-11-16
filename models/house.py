from model_zoo import Model
import tensorflow as tf


class HousePricePredictionModel(Model):
    """
    HousePricePredictionModel
    """
    
    def inputs(self):
        """
        Define input shape.
        :return:
        """
        return tf.keras.Input(shape=(13))
    
    def outputs(self, inputs):
        """
        Build model.
        :param inputs:
        :return:
        """
        return tf.keras.layers.Dense(1)(inputs)
