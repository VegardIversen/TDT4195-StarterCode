import utils
import skimage
import skimage.morphology
import numpy as np
import matplotlib.pyplot as plt


def remove_noise(im: np.ndarray) -> np.ndarray:
    """
        A function that removes noise in the input image.
        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions


    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(im)
    plt.axis('off')
    plt.title("original")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)


    # axs[0].imshow(im)
    # imgplot = plt.imshow(im)

    print(np.ones((3,3)))

    im = skimage.morphology.binary_closing(im, selem=skimage.morphology.selem.disk(8))
    im = skimage.morphology.binary_erosion(im, selem=skimage.morphology.selem.disk(8))
    im = skimage.morphology.binary_dilation(im, selem=skimage.morphology.selem.disk(2))
    # im = skimage.morphology.binary_erosion(im, selem=np.ones((8, 8))).astype(np.int)
    # im = skimage.morphology.binary_erosion(im, selem=np.ones((8, 8))).astype(np.int)
    # im = skimage.morphology.binary_dilation(im, selem=np.ones((8, 8))).astype(np.int)
    # im = skimage.morphology.binary_closing(im, selem=np.ones((8, 8))).astype(np.int)
    # im = skimage.morphology.area_closing(im, area_threshold=200)
    # im = skimage.morphology.
    # axs[1].imshow(im)

    # showing image
    plt.imshow(im)
    plt.axis('off')

    plt.title("prosessed")

    # plt.show()
    # plt.show()
    return im
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)

    assert im.shape == noise_free_image.shape, "Expected image shape ({}) to be same as resulting image shape ({})".format(
        im.shape, noise_free_image.shape)
    assert noise_free_image.dtype == np.bool, "Expected resulting image dtype to be np.bool. Was: {}".format(
        noise_free_image.dtype)

    noise_free_image = utils.to_uint8(noise_free_image)
    utils.save_im("noisy-filtered.png", noise_free_image)
