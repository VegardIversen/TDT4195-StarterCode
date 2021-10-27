import matplotlib.pyplot as plt
import pathlib
from utils import read_im, save_im
import numpy as np
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)



im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    grey = np.array([ 0.212, 0.7152, 0.0722]) #gray from the assignment
    #here I dont need to have the slicing but its nice to have if there is a RGBa image.
    im = np.dot(im[:,:,:3],grey)

    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")
#plt.show()


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    im = -im + 255
    return im

im_greyscale_inverse = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_greyscale_inverse.jpg"), im_greyscale_inverse, cmap="gray")
plt.imshow(im_greyscale_inverse, cmap="gray")
#plt.show()