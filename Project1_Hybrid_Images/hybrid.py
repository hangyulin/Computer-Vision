import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # m is height, n is width
    kernel_m_size = kernel.shape[0]
    kernel_n_size = kernel.shape[1]
    return_image = np.zeros(img.shape)
    img = img.astype(np.float32)
    # if kernel larger than image
    diff_m = max(0, kernel_m_size - img.shape[0])
    diff_n = max(0, kernel_n_size - img.shape[1])
    
    if len(return_image.shape) == 2:
        for w in range(img.shape[1]):
            for h in range(img.shape[0]):
                left = max(w - kernel_n_size//2, 0)
                kernel_left = left - (w - kernel_n_size//2)
                right = min(w + kernel_n_size//2, return_image.shape[1]) + 1
                if diff_n == 0:
                    kernel_right = kernel_n_size - max(0, kernel_n_size - (img.shape[1] - left))
                else:
                    kernel_right = kernel_n_size - max(0, kernel_n_size - (img.shape[1] - max(w - kernel_n_size//2 + diff_n, 0))) + diff_n
                top = max(h - kernel_m_size//2, 0)
                kernel_top = top - (h - kernel_m_size//2)
                bottom = min(h + kernel_m_size//2, return_image.shape[0]) + 1
                if diff_m == 0:
                    kernel_bottom = kernel_m_size - max(0, kernel_m_size - (img.shape[0] - top))
                else:
                    kernel_bottom = kernel_m_size - max(0, kernel_m_size - (img.shape[0] - max(0, h - kernel_m_size//2 + diff_m))) + diff_m
                img_part = img[top:bottom, left:right]
                temp_kernel = kernel[kernel_top:kernel_bottom, kernel_left:kernel_right]
                return_image[h][w] = np.sum(np.multiply(img_part, temp_kernel))
    else:
        for w in range(img.shape[1]):
            for h in range(img.shape[0]):
                left = max(w - kernel_n_size//2, 0)
                kernel_left = left - (w - kernel_n_size//2)
                right = min(w + kernel_n_size//2, return_image.shape[1]) + 1
                if diff_n == 0:
                    kernel_right = kernel_n_size - max(0, kernel_n_size - (img.shape[1] - left))
                else:
                    kernel_right = kernel_n_size - max(0, kernel_n_size - (img.shape[1] - max(w - kernel_n_size//2 + diff_n, 0))) + diff_n
                top = max(h - kernel_m_size//2, 0)
                kernel_top = top - (h - kernel_m_size//2)
                bottom = min(h + kernel_m_size//2, return_image.shape[0]) + 1
                if diff_m == 0:
                    kernel_bottom = kernel_m_size - max(0, kernel_m_size - (img.shape[0] - top))
                else:
                    kernel_bottom = kernel_m_size - max(0, kernel_m_size - (img.shape[0] - max(0, h - kernel_m_size//2 + diff_m))) + diff_m
                temp_kernel = kernel[kernel_top:kernel_bottom, kernel_left:kernel_right]
                for c in range(3):
                    img_part = img[top:bottom, left:right, c]
                    return_image[h,w,c] = np.sum(np.multiply(img_part, temp_kernel))
        return_image = return_image
    return return_image

## Mistake here, forgot to put kernel_flip into the cross_correlation, I put the oringal kernel instead
def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel_flip = np.fliplr(np.flipud(kernel))
    return cross_correlation_2d(img, kernel_flip)

## Needed to put float() around the integers for python 2
## I used the formula from the slides for the gaussian,
## but the test.py required to divide by total weight and not 1/(2*np.pi*np.square(sigma))
def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    sigma = float(sigma)
    return_kernel = np.zeros((height, width))
    w_m = width//2
    h_m = height//2
    for i in range(height):
        for j in range(width):
            w = float(w_m - j)
            h = float(h_m - i)
            return_kernel[i][j] = np.e ** (-((w**2 + h**2))/(2* sigma**2))
    return_kernel = return_kernel/np.sum(return_kernel)
    return return_kernel
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

