import numpy as np
from PIL import Image
import os
import collections

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

"""set = {name: {tag: [[]] , image: [[]], face: [[]],
                scaled_tag: [[]], box: [[]], bbx_x_len: int, bbx_y_len: int,"""

TrainImage = collections.namedtuple("TrainImage", ["curr_pixels", "curr_est_shape", "curr_addition",
                                                   "regressor_addition", "true_shape", "face_img"])

T = 5   # amount of regressors (forests) in the cascade
K = 10  # amount of trees in the forest
learning_rate_param = 0.8

Shrinkage_factor = 0.1
Tree_Amount = 20
Trees_Depth = 4
Pool_size = 400


def generate_rand_centered_pixels_by_mean(mean_shape):
    max_y, min_y, max_x, min_x = find_bounding_box(mean_shape)
    rand_x = np.random.choice(max_x, Pool_size)
    rand_y = np.random.choice(max_y, Pool_size)

    points = np.array([rand_x, rand_y])

    return points.transpose()


def find_bounding_box(shape):
    max_y = np.max(shape[:][1])
    min_y = np.min(shape[:][1])
    max_x = np.max(shape[:][0])
    min_x = np.min(shape[:][0])

    return max_y, min_y, max_x, min_x


def scale_shape_to_percentage(min_x, min_y, bbx_x_len, bbx_y_len, shape):
    ## TODO: maybe should be changed center and not working
    scaled_shape = np.array(shape)
    scaled_shape[:, 0] = (scaled_shape[:, 0] - min_x) / bbx_x_len
    scaled_shape[:, 1] = (scaled_shape[:, 1] - min_y) / bbx_y_len
    return np.array(scaled_shape)


def scale_all(set_dict):

    for name in set_dict.keys():
        shape = set_dict[name]["tag"]

        max_y, min_y, max_x, min_x = find_bounding_box(shape)
        bbx_x_len = max_x - min_x
        bbx_y_len = max_y - min_y
        scaled_shape = scale_shape_to_percentage(min_x, min_y, bbx_x_len, bbx_y_len, shape)
        box = [[max_y, min_x], [min_y, min_x], [min_y, max_x], [max_x, max_y]]

        set_dict[name]["scaled_tag"] = scaled_shape
        set_dict[name]["box"] = box
        set_dict[name]["bbx_x_len"] = bbx_x_len
        set_dict[name]["bbx_y_len"] = bbx_y_len
        # set_dict[name]["face"] = set_dict[name]["image"][min_x : max_x + 1, min_y : max_y+1]  TODO: needed but not working with PIL


def calc_mean_shape(scaled_shapes):
    ## TODO: maybe should be changed center
    mean_shape = np.zeros(np.shape(scaled_shapes[0]))

    for shape in scaled_shapes:
        mean_shape += shape

    mean_shape /= len(scaled_shapes)

    return mean_shape


def mean_shape_to_curr_bbx():
    pass


def preprocess_data():
    path = "/Users/hagardolev/Documents/Computer-Science/Seconed-Year/ComputerVision/vision_project"
    train = {}
    test = {}
    all_training_tags = []
    all_testing_tags = []

    for i in range(4):
        curr_path = path + '/train_' + str(i+1)
        for filename in os.listdir(curr_path):
            #im = np.array(Image.open(curr_path + '/' + filename))
            im = Image.open(curr_path + '/' + filename)
            nice_name = filename[:-4]
            train[str(nice_name)] = {"image": im}

    for filename in os.listdir(path + '/annotation'):
        curr_file = open(path + '/annotation/' + filename)
        name = curr_file.readline().strip()
        points = []
        for line in curr_file.readlines():
            x, y = line.split(',')
            x = float(x.strip())
            y = float(y.strip())
            points.append([x, y])
        if name in train.keys():
            train[name]["tag"] = points
            all_training_tags.append(points)
        else:
            test[name] = {"tag": points}
            all_testing_tags.append(points)

    test_path = path + "/test"
    for filename in os.listdir(test_path):
        im = Image.open(test_path + '/' + filename)
        nice_name = filename[:-4]
        if nice_name not in test.keys():
            print("ERRRRR")
        else:
            test[nice_name]["image"] = im

    scale_all(train)
    scale_all(test)
    train = train.values()
    test = test.values()
    all_scaled_training_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in train])
    all_scaled_test_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in test])
    mean_shape = calc_mean_shape(np.array(all_scaled_training_tags))

    return mean_shape, train, test, all_training_tags, all_testing_tags, all_scaled_training_tags, all_scaled_test_tags


def generate_training_data(train, all_scaled_training_tags, mean_shape):
    train_data = []
    if len(all_scaled_training_tags) == 1 or len(all_scaled_training_tags) == 0:
        print("NOPE")
        exit(1)

    extracted_pixels_by_mean_shape = generate_rand_centered_pixels_by_mean(mean_shape)

    for i, sample in enumerate(train):
        initial_shapes_option_probabilities = [1/(len(all_scaled_training_tags) - 1)] * len(all_scaled_training_tags)
        initial_shapes_option_probabilities[i] = 0
        curr_est_shape = np.random.choice(all_scaled_training_tags, p=initial_shapes_option_probabilities)

        curr_pixels = get_scaled_pixels_according_mean_shape(extracted_pixels_by_mean_shape, sample.face_img)
        curr_train_img = TrainImage(curr_pixels, curr_est_shape, np.zeros(curr_est_shape.size), np.zeros(curr_est_shape.size),
                                    sample["tag"], sample["face"])
    pass


def main():
    mean_shape, train, test, all_training_tags, all_testing_tags, all_scaled_training_tags, all_scaled_test_tags = preprocess_data()
    train_data = generate_training_data(train)



if __name__ == '__main__':
    main()