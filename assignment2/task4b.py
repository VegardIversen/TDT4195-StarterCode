import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    #from equation 4 in assignment
    fft_kernel = np.fft.fft2(kernel)
    fft_kernel = np.fft.fftshift(fft_kernel)
    fft = np.fft.fft2(im)
    filt_fft_im = fft * fft_kernel
    conv_result = np.fft.ifft2(filt_fft_im).real

    #shifting to center the spectrum

    fft = np.abs(np.fft.fftshift(fft))
    #fft = amplitude(fft)
    filt_fft_im = np.abs(np.fft.fftshift(filt_fft_im))

    #filt_fft_im = np.log(filt_fft_im + 1)
    #fft = np.log(fft + 1)
    #filt_fft_im = amplitude(filt_fft_im)
    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        # Visualize FFT
        plt.imshow(fft)
        plt.subplot(1, 5, 3)
        # Visualize FFT kernel
        plt.imshow(fft_kernel)
        plt.subplot(1, 5, 4)
        # Visualize filtered FFT image
        plt.imshow(filt_fft_im)
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")
        

        #if the kernel[0][0] is 0 its a lowpass filter, and highpass if 1
        if (fft_kernel[0][0] == 1.0):
            plt.savefig(utils.image_output_dir.joinpath("task4b_highpass.png"))
        elif (fft_kernel[0][0] == 0.0):
            plt.savefig(utils.image_output_dir.joinpath("task4b_lowpass.png"))
        else:
            print("Unknown type")


    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
