from utils import *


class Node(object):

    def __init__(self, curr_depth, max_depth, pool_size): # images):
        self.left = None
        self.right = None

        self.curr_depth = curr_depth
        self.max_depth = max_depth

        self.pool_size = pool_size

        self.condition = None

        self.leaf = False
        self.delta_sum = None  ## Should be shape size
        self.right_delta_average = None ## Should be shape size
        self.left_delta_average = None  ## Should be shape size

        if curr_depth == max_depth:
            self.leaf = True

    def generate_rand_cond(self):
        pixel_a, pixel_b = np.random.choice(range(self.pool_size), 2)
        threshold = np.random.random() * Intensity_Change_Threshold

        return Condition(pixel_a, pixel_b, threshold)

    def calc_lserror(self, cond, images):
        sum_average_right = 0
        sum_average_left = 0
        right_imgs = []
        left_imgs = []
        error = 0

        for image in images:
            # print(image.curr_pixels)
            if image.curr_pixels[cond.pixel_a_loc] - image.curr_pixels[cond.pixel_b_loc] > cond.threshold:
                sum_average_right += (image.true_shape - image.curr_est_shape)
                right_imgs.append(image)
            else:
                sum_average_left += (image.true_shape - image.curr_est_shape)
                left_imgs.append(image)
        if len(right_imgs) == 0:
            sum_average_right = 0
            sum_average_left = sum_average_left/len(left_imgs)
        elif len(left_imgs) == 0:
            sum_average_left = 0
            sum_average_right = sum_average_right/len(right_imgs)

        else:
            sum_average_right = sum_average_right / len(right_imgs)
            sum_average_left = sum_average_left/len(left_imgs)

        for image in right_imgs:
            error += np.linalg.norm(image.true_shape - sum_average_right) ## Euclidian dist
        for image in left_imgs:
            error += np.linalg.norm(image.true_shape - sum_average_left) ## Euclidian dist

        return error

    def randomize_and_choose_cond(self, images):
        if len(images) == 0:
            print("why like this")
            self.condition = self.generate_rand_cond()
            return self.condition, [], []

        curr_cond = self.generate_rand_cond()
        curr_err = self.calc_lserror(curr_cond, images)
        min_err = curr_err
        min_cond = curr_cond

        for i in range(Amount_of_Rand_conditions):
            curr_cond = self.generate_rand_cond()
            curr_err = self.calc_lserror(curr_cond, images)
            if curr_err < min_err:
                min_err = curr_err
                min_cond = curr_cond


        right_imgs = []
        left_imgs = []

        for image in images:
            if image.curr_pixels[min_cond.pixel_a_loc] - image.curr_pixels[min_cond.pixel_b_loc] > min_cond.threshold:
                right_imgs.append(image)
            else:
                left_imgs.append(image)

        # self.right_imgs = right_imgs
        # self.left_imgs = left_imgs
        self.condition = min_cond

        return min_cond, right_imgs, left_imgs

    def set_leaf_params(self, cond, right_imgs, left_imgs):
        sum_average_right = 0
        sum_average_left = 0
        right_imgs_count = len(right_imgs)
        left_imgs_count = len(left_imgs)
        if (left_imgs_count + right_imgs_count) == 0:
            print("whyyyyyy 2")
            self.right_delta_average = 0
            self.left_delta_average = 0
            self.condition = cond
            return

        for img in right_imgs:
            sum_average_right += (img.true_shape - img.curr_est_shape)
        for img in left_imgs:
            sum_average_left += (img.true_shape - img.curr_est_shape)

        if len(right_imgs) == 0:
            sum_average_right = 0
            sum_average_left = sum_average_left/len(left_imgs)
        elif len(left_imgs) == 0:
            sum_average_left = 0
            sum_average_right = sum_average_right/len(right_imgs)

        else:
            sum_average_right = sum_average_right/right_imgs_count
            sum_average_left = sum_average_left/left_imgs_count

        self.right_delta_average = sum_average_right
        self.left_delta_average = sum_average_left
        self.condition = cond

        for img in right_imgs:
            img.curr_addition = self.right_delta_average
        for img in left_imgs:
            img.curr_addition = self.left_delta_average

    def train(self, images):
        self.condition, right_imgs, left_imgs = self.randomize_and_choose_cond(images)

        if self.leaf:
            self.set_leaf_params(self.condition, right_imgs, left_imgs)
            return

        self.right = Node(self.curr_depth + 1, self.max_depth, self.pool_size)
        self.left = Node(self.curr_depth + 1, self.max_depth, self.pool_size)

        self.right.train(right_imgs)
        self.left.train(left_imgs)

    def predict(self, image):
        if self.leaf:
            if image.curr_pixels[self.condition.pixel_a_loc] - image.curr_pixels[self.condition.pixel_b_loc] > self.condition.threshold:
                return self.right_delta_average
            return self.left_delta_average

        else:
            if image.curr_pixels[self.condition.pixel_a_loc] - image.curr_pixels[self.condition.pixel_b_loc] > self.condition.threshold:
                return self.right.predict()
            return self.left.predict()


class Tree(object):
    def __init__(self, depth, pool_size):
        self.root = None
        self.depth = depth
        self.pool_size = pool_size

    def train(self, images):
        """

        :param images: of the type [ TrainImage, ..., TrainImage]
        "TrainImage", ["curr_pixels", "curr_est_shape", "curr_addition", "true_shape", "face_img"]
        :return:
        """
        self.root = Node(0, self.depth, self.pool_size)
        self.root.train(images)
        print("Finished training tree")

    def predict_delta(self, image):
        return self.root.predict(image)

    def predict_deltas_images(self, images):
        predictions = []
        for image in images:
            predictions.append(self.root.predict(image))

        return predictions

