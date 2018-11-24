import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread as imread
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

GS_REP = 1
RGB_REP = 2
MAX_VALUE = 256


def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    im = imread(filename)
    if representation == GS_REP:
        im = rgb2gray(im)
    im = np.divide(im, MAX_VALUE - 1)
    return im


def reg_to_hom(points):
    """
    Converts point in [x, y] coordinates to homogenous coordinates
    :param points: an array of shape (n, 2)
    :return: array of shape (n, 3) with the points converted to homogenous ones
    """
    return np.hstack((points, np.ones((points.shape[0], 1))))


def hom_to_reg(points):
    """
    Converts point in homogenous coordinates to [x, y] coordinates
    :param points: an array of shape (n, 3)
    :return: array of shape (n, 2) with the points converted to regular coordinates
    """
    points_z = points[:, 2]
    reg_x = np.divide(points[:, 0], points_z)
    reg_y = np.divide(points[:, 1], points_z)
    return np.vstack((reg_x, reg_y)).transpose()


def left_multiply_vectors(mat, arr):
    """
    Left multiplies each vector in an array of vectors with a matrix.
    :param mat: array of shape (n * n)
    :param arr: array of shape (m * n)
    :return: array of shape (m * n) with the result
    """
    return np.einsum('ij, kj->ki', mat, arr)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1_hom = reg_to_hom(pos1)
    pos2_hom = left_multiply_vectors(H12, pos1_hom)
    return hom_to_reg(pos2_hom)


def normalize_hom(H):
    """
    Normalizes a (3 * 3) homography to have 1 in the lower-left cell
    :param H: array of shape (3 * 3)
    :return: array of shape (3 * 3)
    """
    H /= H[2, 2]
    return H


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the upper left corner,
     and the second row is the [x,y] of the lower right corner
    """
    w_i, h_i = w - 1, h - 1
    points = np.array([[0, 0], [w_i, 0], [0, h_i], [w_i, h_i]])

    points = reg_to_hom(points)
    new_points = left_multiply_vectors(homography, points)
    new_points = np.round(hom_to_reg(new_points)).astype(int)

    new_points = np.sort(new_points, axis=0)
    new_top_left = new_points[0]
    new_bottom_right = new_points[new_points.shape[0] - 1]

    return np.vstack((new_top_left, new_bottom_right))


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography.
    :return: A 2d warped image.
    """
    h, w = image.shape
    corners = compute_bounding_box(homography, w, h)

    x_range = np.arange(corners[0][0], corners[1][0] + 1)
    y_range = np.arange(corners[0][1], corners[1][1] + 1)
    x_i, y_i = np.meshgrid(x_range, y_range)
    rows, cols = x_i.shape

    grid = np.dstack((x_i, y_i)).reshape(rows * cols, 2)
    inv_hom = np.linalg.inv(homography)
    grid_t = apply_homography(grid, inv_hom).reshape(rows, cols, 2)

    x_i_t, y_i_t = grid_t[:, :, 0], grid_t[:, :, 1]

    warped_image = map_coordinates(image, [y_i_t, x_i_t], order=1, prefilter=False)
    return warped_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])
\
def display_matches(im1, im2, points1, points2):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    full_im = np.hstack((im1, im2))
    plt.imshow(full_im, cmap='gray')

    s_points2 = points2.copy()
    s_points2[:, 0] += im1.shape[1]

    # plotting points
    full_points = np.vstack((points1, s_points2))
    plt.plot(full_points[:, 0], full_points[:, 1], 'ro', markersize=1.5)

    # # preparing inliers
    # m = points1.shape[0]
    # is_inlier = np.zeros(m).astype(bool)
    # is_inlier[inliers] = True
    #
    # # plotting lines
    # plot_lines(points1, s_points2, ~is_inlier, "b")
    # plot_lines(points1, s_points2, is_inlier, "y", .6)

    plt.show()
