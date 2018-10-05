from model import BostonHousingModel
import tensorflow as tf
from model_zoo.trainer import BaseTrainer

tf.flags.DEFINE_integer('epochs', 20, 'Max Epochs')


class Trainer(BaseTrainer):
    
    def __init__(self):
        BaseTrainer.__init__(self)
        self.model_class = BostonHousingModel
    
    def prepare_data(self):
        from tensorflow.python.keras.datasets import boston_housing
        from sklearn.preprocessing import StandardScaler
        (x_train, y_train), (x_eval, y_eval) = boston_housing.load_data()
        ss = StandardScaler()
        ss.fit(x_train)
        x_train, x_eval = ss.transform(x_train), ss.transform(x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
