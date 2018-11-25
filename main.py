import sys
from utils import *
from data_preperation import *
from regressors import Regressor

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def train_model(train_data, target_path, extracted_pixels_by_mean_shape, mean_shape):
    regressors = []
    for i in range(Cascades):
        curr_regr = Regressor(mean_shape, extracted_pixels_by_mean_shape)
        curr_regr.train(train_data)
        regressors.append(curr_regr)
        print("Finished regressor number 1")

    with open(target_path, 'wb') as target:
        pickle.dump(regressors, target)


def test_model(test_data, train_model_save_path):
    model = pickle.load(train_model_save_path)
    error = 0
    for image in test_data:
        model.predict(image)



def main():
    if len(sys.argv) < 3:
        print("Give me dataaaaa and model save pathhhh")
        exit(-1)
    path = sys.argv[1]
    train_model_save_path = sys.argv[2]

    mean_shape, train, test, all_true_train_shapes = preprocess_data(path)
    print("finished_preprocessing")
    train_data, extracted_pixels_by_mean_shape = generate_training_data(train, all_true_train_shapes, mean_shape)
    print("Finished training data pre process")
    train_model(train_data, train_model_save_path, extracted_pixels_by_mean_shape, mean_shape)
    print("Finished training")

    test_model(test, train_model_save_path)

if __name__ == '__main__':
    main()