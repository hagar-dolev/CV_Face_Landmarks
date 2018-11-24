import numpy as np
from tree import Tree, TrainImage

'"TrainImage", ["curr_pixels", "curr_est_shape", "curr_addition", "true_shape", "face_img"]'

Shrinkage_factor = 0.1
Tree_Amount = 20
Trees_Depth = 4
Pool_size = 400


def calc_transform(mean_shape, curr_est_shape):
    pass


def get_new_pixels(pixel_loc_in_meanShape, other_shape, face_img):
    transformation = calc_transform(pixel_loc_in_meanShape, other_shape)
    return 1


class Regressor(object):

    def __init__(self, mean_shape):
        self.mean_shape = mean_shape
        self.pixel_loc_in_meanShape = np.zeros((Pool_size, 2))
        self.trees = []

        for i in range(Tree_Amount):
            self.trees.append(Tree(Trees_Depth, Pool_size))

    def train(self, train_imgs):

        for tree in self.trees:
            tree.train(train_imgs)
            for img in train_imgs:
                # transfor_operator = calc_transform(self.mean_shape, img.curr_est_shape)

                img.curr_est_shape += Shrinkage_factor * img.curr_addition
                img.regressor_addition += Shrinkage_factor * img.curr_addition  # Might not be necessary

                img.curr_addition = np.zeros(len(self.mean_shape))

                img.curr_pixels = get_new_pixels(self.pixel_loc_in_meanShape, img.curr_est_shape, img.face_img)




