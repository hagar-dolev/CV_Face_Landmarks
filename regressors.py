from tree import Tree
from utils import *


# def calc_transform(mean_shape, curr_est_shape):
#     pass


# def get_new_pixels(pixel_loc_in_meanShape, other_shape, face_img):
#     indicises = [scale_shape(np.shape(face_img), pixel_loc_in_meanShape)[:, 1],
#                  scale_shape(np.shape(face_img), pixel_loc_in_meanShape)[:, 0]]
#     # transformation = calc_transform(pixel_loc_in_meanShape, other_shape)
#     curr_pixels = face_img[indicises]
#
#     return curr_pixels


class Regressor(object):

    def __init__(self, mean_shape, extracted_pixels_by_mean_shape):
        self.mean_shape = mean_shape
        self.pixel_loc_in_meanShape = extracted_pixels_by_mean_shape
        self.trees = []

        for i in range(Tree_Amount):
            self.trees.append(Tree(Trees_Depth, Pool_size))

    def train(self, train_imgs):
        print(np.shape(train_imgs))
        for tree in self.trees:
            tree.train(train_imgs)
            for img in train_imgs:
                # transfor_operator = calc_transform(self.mean_shape, img.curr_est_shape)
                img.curr_est_shape = img.curr_est_shape.astype(np.float64)
                img.regressor_addition = img.regressor_addition.astype(np.float64)

                img.curr_est_shape += Shrinkage_factor * img.curr_addition
                # img.regressor_addition += Shrinkage_factor * img.curr_addition  # Might not be necessary

                img.curr_addition = np.zeros(len(self.mean_shape))

                # img.curr_pixels = get_new_pixels(self.pixel_loc_in_meanShape, img.curr_est_shape.astype(int), img.face_img)

    def predict_one(self, sample):
        curr_est_shape = scale_shape(np.shape(sample.face), self.mean_shape).astype(np.float64)
        indicises = [scale_shape(np.shape(sample.face), self.pixel_loc_in_meanShape)[:, 1], scale_shape(np.shape(sample.face), self.pixel_loc_in_meanShape)[:, 0]]
        if sample.face.shape[0] == 0 or sample.face.shape[1] == 0:
            pass
        else:
            curr_pixels = sample.face[indicises]
            curr_train_img = TrainImage(curr_pixels, curr_est_shape, np.zeros(curr_est_shape.size),
                                        np.zeros(curr_est_shape.size),
                                        sample.true_shape, sample.face)
            for tree in self.trees:
                curr_est_shape += tree.predict_delta(curr_train_img).astype(np.float64)

        return curr_est_shape

    # def predict_many(self, faces):
    #     test_samples = []
    #     for face in faces:
    #         curr_est_shape = scale_shape(np.shape(face), self.mean_shape)
    #         indicises = [scale_shape(np.shape(face), self.pixel_loc_in_meanShape)[:,1], scale_shape(np.shape(face), self.pixel_loc_in_meanShape)[:,0]]
    #         if face.shape[0] == 0 or face.shape[1] == 0:
    #             pass
    #         else:
    #             curr_pixels = face[indicises]
    #             curr_train_img = TrainImage(curr_pixels, curr_est_shape, np.zeros(curr_est_shape.size),
    #                                         np.zeros(curr_est_shape.size),
    #                                         face.true_shape, face.face)
    #             test_samples.append(curr_train_img)
    #
    #     for tree in self.trees:
    #         delta_additions = tree.predict_delta(test_samples)
    #         for face, delta in enumerate(test_samples, delta_additions):
    #             face.curr_est_shape += delta
    #
    #     return
