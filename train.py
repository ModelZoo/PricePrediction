from model_zoo.trainer import BaseTrainer
from model_zoo.preprocess import standardize
from model_zoo import flags, datasets

flags.DEFINE_integer('epochs', 100, 'Max epochs')
flags.DEFINE_string('model_class_name', 'HousePricePredictionModel', 'Model class name')


class Trainer(BaseTrainer):
    """
    Train Price Prediction Model.
    """

    def prepare_data(self):
        """
        Prepare train data.
        :return:
        """
        (x_train, y_train), (x_eval, y_eval) = datasets.boston_housing.load_data()
        x_train, x_eval = standardize(x_train, x_eval)
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
