import collections
import numpy as np
# class FaceImg(object):
#     def __init__(self, image, curr_shape):

Condition = collections.namedtuple("Condition", ["pixel_a_loc", "pixel_b_loc", "threshold"])
TrainImage = collections.namedtuple("TrainImage", ["curr_pixels", "curr_est_shape", "curr_addition", "true_shape", "face_img"])
Intensity_Change_Threshold = 100
Amount_of_Rand_conditions = 100

class Node(object):

    def __init__(self, curr_depth, max_depth, pixels_pool): # images):
        self.left = None
        self.right = None

        self.curr_depth = curr_depth
        self.max_depth = max_depth

        # self.images = images
        self.pixels_pool = pixels_pool

        self.condition = None

        # self.parent = None

        # self.right_imgs = None
        # self.left_imgs = None

        self.leaf = False
        self.delta_sum = None  ## Should be shape size
        self.right_delta_average = None ## Should be shape size
        self.left_delta_average = None  ## Should be shape size

        if curr_depth == max_depth:
            self.leaf = True


    def set_right_child(self):
        pass
    def set_left_child(self):
        pass

    def generate_rand_cond(self):
        pixel_a, pixel_b = np.random.choice(self.pixels_pool, 2)
        threshold = np.random.random() * Intensity_Change_Threshold

        return Condition(pixel_a, pixel_b, threshold)


    def calc_lserror(self, cond, images):
        sum_average_right = 0
        sum_average_left = 0
        right_imgs = []
        left_imgs = []
        error = 0

        for image in images:
            if image.curr_pixels[cond.pixel_a_loc] - image.curr_pixels[cond.pixel_b_loc] > cond.threshold:
                sum_average_right += (image.true_shape - image.curr_est_shape)
                right_imgs.append(image)
            else:
                sum_average_left += (image.true_shape - image.curr_est_shape)
                left_imgs.append(image)

        sum_average_right /= len(right_imgs)
        sum_average_left /= len(left_imgs)

        for image in right_imgs:
            error += np.linalg.norm(image.true_shape - sum_average_right) ## Euclidian dist
        for image in left_imgs:
            error += np.linalg.norm(image.true_shape - sum_average_left) ## Euclidian dist

        return error



    def randomize_and_choose_cond(self, images):
        curr_cond = self.generate_rand_cond()
        curr_err = self.calc_lserror(curr_cond)
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


    def set_leaf_params(self, cond, images):
        sum_average_right = 0
        sum_average_left = 0
        right_imgs_count = 0
        left_imgs_count = 0

        for image in images:
            if image.curr_pixels[cond.pixel_a_loc] - image.curr_pixels[cond.pixel_b_loc] > cond.threshold:
                sum_average_right += (image.true_shape - image.curr_est_shape)
                right_imgs_count += 1
            else:
                sum_average_left += (image.true_shape - image.curr_est_shape)
                left_imgs_count += 1

        sum_average_right /= right_imgs_count
        sum_average_left /= left_imgs_count

        self.right_delta_average = sum_average_right
        self.left_delta_average = sum_average_left
        self.condition = cond


    def train(self, images):
        self.condition, right_imgs, left_imgs = self.randomize_and_choose_cond(images)

        if self.leaf:
            self.set_leaf_params(self.condition, images)
            return

        self.right = Node(self.curr_depth + 1, self.max_depth, self.pixels_pool)
        self.left = Node(self.curr_depth + 1, self.max_depth, self.pixels_pool)

        self.right.train(right_imgs)
        self.left.train(left_imgs)

        # self.images = images

    # def train(self, pixels, shape_delta):
    #     self.count_images += 1
    #     if not self.leaf:
    #         if pixels[self.pixel_index1] - pixels[self.pixel_index2] > self.threshold:
    #             self.right.train(pixels, shape_delta)
    #         else:
    #             self.left.train(pixels, shape_delta)
    #     else:
    #         self.delta_sum += shape_delta
    #         self.delta_average = self.delta_sum / self.count_images
    #
    #
    # def predict(self, pixels):
    #     if not self.leaf:
    #         if pixels[self.pixel_index1] - pixels[self.pixel_index2] > self.threshold:
    #             return self.right.predict(pixels)
    #         else:
    #             return self.left.predict(pixels)
    #     return self.delta_average


class Tree(object):
    def __init__(self, depth):
        self.root = None
        self.depth = depth


    def build(self, pixel_indicises, ):