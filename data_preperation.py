from utils import *
from PIL import Image
import os

"""set = {name: {tag: [[]] , image: [[]], face: [[]],
                scaled_tag: [[]], box: [[]], bbx_x_len: int, bbx_y_len: int,"""

'Sample = collections.namedtuple("Sample", ["name", "face", "true_shape"])'

#
# def scale_shape_to_percentage(min_x, min_y, bbx_x_len, bbx_y_len, shape):
#     ## TODO: maybe should be changed center and not working
#     scaled_shape = np.array(shape)
#     scaled_shape[:, 0] = (scaled_shape[:, 0] - min_x) / bbx_x_len
#     scaled_shape[:, 1] = (scaled_shape[:, 1] - min_y) / bbx_y_len
#     return np.array(scaled_shape)
#
#
# def scale_all(set_dict):
#
#     for name in set_dict.keys():
#         shape = set_dict[name]["tag"]
#
#         max_y, min_y, max_x, min_x = find_bounding_box(shape)
#         bbx_x_len = max_x - min_x
#         bbx_y_len = max_y - min_y
#         scaled_shape = scale_shape_to_percentage(min_x, min_y, bbx_x_len, bbx_y_len, shape)
#         box = [[max_y, min_x], [min_y, min_x], [min_y, max_x], [max_x, max_y]]
#
#         set_dict[name]["scaled_tag"] = scaled_shape
#         set_dict[name]["box"] = box
#         set_dict[name]["face"] = set_dict[name]["image"][min_x : max_x + 1, min_y : max_y+1]


def preprocess_data(path):
    train_dict = {}
    test_dict = {}
    all_training_tags = []
    all_testing_tags = []
    train = []
    test = []

    for i in range(4):
        if i == 0:
            curr_path = path + '/test'
            for filename in os.listdir(curr_path):
                im = read_image(curr_path + '/' + filename, GS_REP)
                nice_name = filename[:-4]
                test_dict[str(nice_name)] = {"image": im}

        curr_path = path + '/train_' + str(i+1)
        for filename in os.listdir(curr_path):
            #im = np.array(Image.open(curr_path + '/' + filename))
            im = read_image(curr_path + '/' + filename, GS_REP)
            nice_name = filename[:-4]
            train_dict[str(nice_name)] = {"image": im}

    for filename in os.listdir(path + '/annotation'):
        curr_file = open(path + '/annotation/' + filename)
        name = curr_file.readline().strip()
        points = []
        for line in curr_file.readlines():
            x, y = line.split(',')
            x = float(x.strip())
            y = float(y.strip())
            points.append([x, y])
        if name in train_dict.keys():
            train_dict[name]["true_shape"] = points
            all_training_tags.append(points)

            train.append(Sample(name, get_face(train_dict[name]["image"], points), points))

        else:
            test_dict[name] = {"true_shape": points}
            all_testing_tags.append(points)
            test.append(Sample(name, get_face(test_dict[name]["image"], points), points))


    # scale_all(train_dict)
    # scale_all(test_dict)

    # all_scaled_training_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in train_dict])
    # all_scaled_test_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in test_dict])
    all_true_train_shapes = np.array([(lambda x: x.true_shape)(x) for x in train])
    mean_shape = calc_mean_shape(np.array(all_true_train_shapes))

    return mean_shape, train, test, all_training_tags, all_testing_tags, all_true_train_shapes


def generate_training_data(train, all_true_train_shapes, mean_shape):
    train_data = []
    if len(all_true_train_shapes) == 1 or len(all_true_train_shapes) == 0:
        print("NOPE")
        exit(1)

    extracted_pixels_by_mean_shape = generate_rand_centered_pixels_by_mean(mean_shape)

    for i, sample in enumerate(train):
        initial_shapes_option_probabilities = [1/(len(all_true_train_shapes) - 1)] * len(all_true_train_shapes)
        initial_shapes_option_probabilities[i] = 0
        curr_est_shape = np.random.choice(all_true_train_shapes, p=initial_shapes_option_probabilities)
        curr_est_shape = scale_shape(curr_est_shape, np.shape(sample.face))

        curr_pixels = get_scaled_pixels_according_mean_shape(extracted_pixels_by_mean_shape, curr_est_shape, sample.face_img)
        curr_train_img = TrainImage(curr_pixels, curr_est_shape, np.zeros(curr_est_shape.size), np.zeros(curr_est_shape.size),
                                    sample.true_shape, sample.face)
    pass