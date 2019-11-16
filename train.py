from model_zoo.trainer import BaseTrainer
from model_zoo.preprocess import standardize
from model_zoo import flags, datasets

flags.define('epochs', 100)
flags.define('model_class_name', 'HousePricePredictionModel')
flags.define('checkpoint_name', 'model.h5')
flags.define('checkpoint_save_weights_only', False)


class Trainer(BaseTrainer):
    """
    Train Price Prediction Model.
    """
    
    def data(self):
        """
        Prepare train data.
        :return:
        """
        (x_train, y_train), (x_eval, y_eval) = datasets.boston_housing.load_data()
        x_train, x_eval = standardize(x_train, x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return self.generator(*train_data), eval_data


if __name__ == '__main__':
    Trainer().run()
