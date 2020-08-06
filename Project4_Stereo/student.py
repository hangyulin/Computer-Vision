import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    images_arr = np.array(images)

    if images_arr.shape[3] == 1:
        normals = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype = np.float32)
    else:
        normals = np.zeros(images[0].shape, dtype = np.float32)

    l_0 = np.linalg.inv(lights.T.dot(lights))
    
    g_0 = np.einsum('ij, jklm -> iklm', lights.T, images_arr)
    g = np.einsum('ij, jklm -> iklm', l_0, g_0)
    albedo = np.linalg.norm(g, axis = 0)

    for i in range(3):
        for j in range(albedo.shape[2]):
            if albedo.shape[2] == 3:
                temp_normals = np.divide(g[i,:,:,j], albedo[:,:,j], out = np.zeros_like(g[i,:,:,j]), where = albedo[:,:,j] != 0)
            else:
                temp_normals = np.divide(g[i,:,:,0], albedo[:,:,0], out = np.zeros_like(g[i,:,:,0]), where = albedo[:,:,0] != 0)
            normals[:,:,i] += temp_normals

    normals /= albedo.shape[2]

    return albedo, normals

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # raise NotImplementedError()
    proj_matrix = np.dot(K, Rt)
    h, w, _ = points.shape
    projections = np.zeros([h, w, 2], dtype = np.float32)
    for i in range(h):
        for j in range(w):
            cur_pt = np.ones(4)
            cur_pt[:3] = points[i][j]
            pt = np.dot(proj_matrix, cur_pt)
            pt /= pt[-1]
            projections[i][j] = pt[:-1]
            
    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # raise NotImplementedError()
    height, width, channel = image.shape
    normalized = np.zeros([height, width, channel * ncc_size ** 2])
    patch_size = int(ncc_size / 2)
    
    image_t = image.T
    for h in range(patch_size, height - patch_size):
        patches = []
        for w in range(patch_size, width - patch_size):
            patches.append(image_t[:, w - patch_size: w + patch_size + 1, h - patch_size : h + patch_size + 1])

        patches = np.array(patches)
        
        channel_matrix = np.reshape(np.transpose(patches, (0,1,3,2)), (len(patches), 3, ncc_size **2)).astype(np.float32)

        channel_matrix -= np.mean(channel_matrix, axis = 2, keepdims = True)
            
        for i in range(len(channel_matrix)):
            std = np.linalg.norm(channel_matrix[i])
            if std >= 1e-6:
                normalized[h, patch_size + i] = channel_matrix[i].reshape([-1]) / std

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # raise NotImplementedError()
    # height, width, _ = image1.shape
    # ncc = np.zeros([height, width], dtype = np.float32)
    # for h in range(height):
    #     for w in range(width):
    #         ncc[h, w] = np.dot(image1[h, w], image2[h, w])
    ncc = np.einsum('ijk, ijk -> ij', image1, image2)
    return ncc