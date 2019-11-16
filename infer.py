from model_zoo.inferer import BaseInferer
from model_zoo.preprocess import standardize
from model_zoo import flags, datasets

flags.DEFINE_string('checkpoint_name', 'model-best.h5', help='Model name')


class Inferer(BaseInferer):
    """
    Inferer for House Price Prediction.
    """
    
    def data(self):
        """
        Predict model.
        :return:
        """
        (x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()
        _, x_test = standardize(x_train, x_test)
        return x_test


if __name__ == '__main__':
    result = Inferer().run()
    print(result)
