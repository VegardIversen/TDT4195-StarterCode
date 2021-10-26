import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)

#P = ((S-1)*W-S+F)/2#, with F = filter size, S = stride, W = input size#calculating padding for no odd number and with a stride 
#but here we assume odd number and stride =1 we get that the padding size is P= (F-1)/2
def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))

def convole2D_kxk(matrix, kernel):
    matrix_size = matrix.shape[0]
    #matrix_depth = matrix.shape[2]
    #mat = np.zeros(shape=(matrix_size,matrix_size,matrix_depth))
    pix_val = 0

    #print('---------------matrise----------------')
    #print(matrix)
    for i in range(matrix_size):
        for j in range(matrix_size):
            #print(f'i: {i}, j: {j}')
            pix_val += matrix[j,i]*kernel[j,i]
            #print(pix_val)
    #print('---------------ny matrise---------------')
    #print(mat)

    return pix_val


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """

    assert len(im.shape) == 3
    image_height = im.shape[0]
    image_width = im.shape[1]
    image_depth = im.shape[2]
    #since the kernel is a KxK matrix we only need one of the sizes
    kernel_size = kernel.shape[0]
    #finding the size of the padding
    pad_size = (kernel_size-1)//2
    #creating a empty array with padding
    out_im = np.zeros(shape=(image_height+2*pad_size,image_width+2*pad_size,image_depth)) #this seems to pad it correctly when i show it
    im_conv = np.zeros(shape=(image_height+2*pad_size,image_width+2*pad_size,image_depth))
    image_height_pad = out_im.shape[0]
    image_width_pad = out_im.shape[1]
    kernel_flip = np.flip(kernel)
    #fitting the image in the middle, and now I got padding
    out_im[pad_size:-pad_size,pad_size:-pad_size,:] = im
    print(out_im.shape)
    print('----------------------')
  
    
    kernel_new_size = np.array([kernel]*image_depth)
    #print(im[0:1,0:4,0:4])
    #print(kernel_new_size[0:1,0:2,0:2].shape)
    #print(kernel_new_size)
    
    #print(np.flip(kernel_new_size).shape)
    #im_pad = pad(im,image_height,image_width)
    for d in range(image_depth):
        for y in range(pad_size,image_height_pad-pad_size):
            for x in range(pad_size,image_width_pad-pad_size):
                pix_val = 0.0
                for n in range(kernel.shape[0]):
                    for k in range(kernel.shape[0]):
                        pix_val += out_im[x-pad_size + n,y-pad_size + k,d]*kernel_flip[n,k]
                im_conv[x,y,d] = pix_val
        
    
                #out_im[x,y,d] = convole2D_kxk(out_im[x-pad_size:x+pad_size+1,y-pad_size:y+pad_size+1,d],kernel)

                
    
    im = im_conv[pad_size:-pad_size,pad_size:-pad_size,:]
    #im = im/np.amax(im)
    #im = np.clip(im,0,1)
    #plt.imshow(out_im[pad_size:-pad_size,pad_size:-pad_size,:])
    plt.imshow(im)
    plt.show()
    return im


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = 1 / 256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(
        im_smoothed, np.ndarray),         f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
    assert im_smoothed.shape == im.shape,         f"Expected smoothed im ({im_smoothed.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert im_sobel.shape == im.shape,         f"Expected smoothed im ({im_sobel.shape}" + \
        f"to have same shape as im ({im.shape})"
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()
