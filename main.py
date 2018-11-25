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
    with open(train_model_save_path, 'rb') as pickle_file:
        cascade_regressors = pickle.load(pickle_file)
    error = np.zeros(test_data[0].true_shape.shape)
    for image in test_data:
        image.curr_est_shape = np.zeros(image.true_shape.shape)
        for regressor in cascade_regressors:
            shape_delta = regressor.predict_one(image)
            image.curr_est_shape += shape_delta

        error += image.true_shape - image.curr_est_shape

    error = error/ len(test_data)
    error = np.linalg.norm(error)

    for i in range(5):
        curr_face = test_data[i]
        display_matches(curr_face.face, curr_face.face, curr_face.true_shape, curr_face.curr_est_shape)

    return error



def main():
    if len(sys.argv) < 3:
        print("Give me dataaaaa and model save pathhhh")
        exit(-1)
    path = sys.argv[1]
    train_model_save_path = sys.argv[2]

    mean_shape, train, test, all_true_train_shapes = preprocess_data(path)
    # print("finished_preprocessing")
    # train_data, extracted_pixels_by_mean_shape = generate_training_data(train, all_true_train_shapes, mean_shape)
    # print("Finished training data pre process")
    # train_model(train_data, train_model_save_path, extracted_pixels_by_mean_shape, mean_shape)
    # print("Finished training")

    test_model(test, train_model_save_path)

if __name__ == '__main__':
    main()