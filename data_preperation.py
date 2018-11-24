from utils import *
import cv2 as cv

import os

"""set = {name: { image: [[]]}"""

'Sample = collections.namedtuple("Sample", ["name", "face", "true_shape"])'


def preprocess_data(path):
    train_dict = {}
    test_dict = {}
    all_training_tags = []
    all_testing_tags = []
    train = []
    test = []
    curr_path = ''
    for i in range(1):
        # if i == 0:
        #     curr_path = path + '/test'
        #     for filename in os.listdir(curr_path):
        #         im = read_image(curr_path + '/' + filename, GS_REP)
        #         nice_name = filename[:-4]
        #         test_dict[str(nice_name)] = {"image": im}

        print(curr_path)

        curr_path = path + '/train_' + str(i+1)
        for filename in os.listdir(curr_path):
            im = read_image(curr_path + '/' + filename, GS_REP)
            nice_name = filename[:-4]
            train_dict[str(nice_name)] = {"image": im}

    print("annotation")
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

        # else:
        #     test_dict[name] = {"true_shape": points}
        #     all_testing_tags.append(points)
        #     test.append(Sample(name, get_face(test_dict[name]["image"], points), points))


    # scale_all(train_dict)
    # scale_all(test_dict)

    # all_scaled_training_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in train_dict])
    # all_scaled_test_tags = np.array([(lambda x: x["scaled_tag"])(x) for x in test_dict])
    all_true_train_shapes = np.array([(lambda x: x.true_shape)(x) for x in train])
    mean_shape = calc_mean_shape(np.array(all_true_train_shapes))

    test = train

    return mean_shape, train, test, all_true_train_shapes


def scale_shape(dest_shape, shape_points):
    """
    scales a list of points to a destination shape
    """
    max_p = np.amax(shape_points, axis=0)
    n, m = (dest_shape[0] - 2) / max_p[0], (dest_shape[1] - 2) / max_p[1]
    resized_x = np.trunc(shape_points[:, 0] * n) + 1
    resized_y = np.trunc(shape_points[:, 1] * m) + 1
    resized_x = resized_x[np.newaxis, :].transpose()
    resized_y = resized_y[np.newaxis, :].transpose()
    return np.hstack((resized_x, resized_y))


def get_scaled_pixels_according_mean_shape(orig_shape, dest_shape, orig_samples, dest_im):
    """
    samples the destination image according to a list of points in an original image.
    :param orig_shape: original face shape
    :param dest_shape: destination face shape
    :param orig_samples: list of 2d points in the original image
    :param dest_im: the destination image
    :return: list of 2d points in the destination image
    """
    transform = cv.estimateRigidTransform(orig_shape, dest_shape, True)
    print(transform)
    dest_sample_points = np.dot(orig_samples, transform)
    dest_sample_points = hom_to_reg(dest_sample_points)
    print(dest_sample_points)
    return dest_im[dest_sample_points]


def generate_training_data(train, all_true_train_shapes, mean_shape):
    train_data = []
    if len(all_true_train_shapes) == 1 or len(all_true_train_shapes) == 0:
        print("NOPE")
        exit(1)

    extracted_pixels_by_mean_shape = generate_rand_pixels_by_mean(mean_shape)

    for i, sample in enumerate(train):
        initial_shapes_option_probabilities = [1/(len(all_true_train_shapes) - 1)] * len(all_true_train_shapes)
        initial_shapes_option_probabilities[i] = 0
        curr_est_shape = all_true_train_shapes[np.random.choice(len(all_true_train_shapes), p=initial_shapes_option_probabilities)]

        # display_matches(sample.face, sample.face, curr_est_shape, scale_shape(np.shape(sample.face), curr_est_shape))
        plt.imshow(sample.face)
        plt.show()

        curr_est_shape = scale_shape(np.shape(sample.face), curr_est_shape)

        curr_pixels = get_scaled_pixels_according_mean_shape(mean_shape, curr_est_shape, extracted_pixels_by_mean_shape, sample.face)

        curr_train_img = TrainImage(curr_pixels, curr_est_shape, np.zeros(curr_est_shape.size), np.zeros(curr_est_shape.size),
                                    sample.true_shape, sample.face)

        train_data.append(curr_train_img)

    return train_data