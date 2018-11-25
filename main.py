import sys
from utils import *
from data_preperation import *
from regressors import Regressor

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def train_model(train_data, target_path):
    regressors = []
    tage_shape = ()
    for i in range(Cascades):
        curr_regr = Regressor(tage_shape)
        curr_regr.train(train_data)
        regressors.append(curr_regr)
        print("Finished regressor number 1")

    pickle.dump(regressors, target_path)


def main():
    if len(sys.argv) < 3:
        print("Give me dataaaaa and model save pathhhh")
        exit(-1)
    path = sys.argv[1]
    train_model_save_path = sys.argv[2]

    mean_shape, train, test, all_true_train_shapes = preprocess_data(path)
    print("finished_preprocessing")
    train_data = generate_training_data(train, all_true_train_shapes, mean_shape)
    print("Finished training data pre process")
    train_model(train_data, train_model_save_path)
    print("Finished training")


if __name__ == '__main__':
    # main()

    # file = open("/Users/galzemach/School/0.HUJI/B/Computer Vision/Project/data/annotation/338.txt")
    file = open("/Users/galzemach/School/0.HUJI/B/Computer Vision/Project/data/annotation/336.txt")
    name = file.readline().strip()
    points = []
    for line in file.readlines():
        x, y = line.split(',')
        x = float(x.strip())
        y = float(y.strip())
        points.append([x, y])

    file.close()
    points = np.array(points)

    im = read_image("/Users/galzemach/School/0.HUJI/B/Computer Vision/Project/data/train_1/16542667_1.jpg", GS_REP)

    scaled_points = scale_shape(im.shape, points)
    # display_points(im, points)
    # display_points(im, scaled_points)
    plt.title("<-- scaled             original -->")
    display_matches(im, im, scaled_points, points)
