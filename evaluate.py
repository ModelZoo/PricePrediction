from model_zoo.evaluater import BaseEvaluater
from model_zoo import flags
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

flags.DEFINE_string('checkpoint_name', 'model-20.ckpt', help='Model name')


class Evaluater(BaseEvaluater):
    
    def data(self):
        """
        Evaluate model.
        :return:
        """
        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()
        ss = StandardScaler()
        ss.fit(x_train)
        x_eval = ss.transform(x_eval)
        return x_eval, y_eval


if __name__ == '__main__':
    result = Evaluater().run()
    print(result)
