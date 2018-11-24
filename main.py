
from utils import *
from data_preperation import *


try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def train_Model():
    pass


def main():
    mean_shape, train, test, all_training_tags, all_testing_tags, all_true_train_shapes = preprocess_data()
    train_data = generate_training_data(train)



if __name__ == '__main__':
    main()