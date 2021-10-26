import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
import time
start_time = time.time()
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
    matrix_depth = matrix.shape[2]
    mat = np.zeros(shape=(1,1,matrix_depth))
    t = np.zeros(shape=matrix_depth)
    
    

    #print('---------------matrise----------------')
    #print(matrix)
    for i in range(matrix_size):
        for j in range(matrix_size):
            #mat[0,0,:] += matrix[j,i,:]*kernel[j,i,:]
            t += matrix[j,i,:]*kernel[j,i,:]
    #         print(f't: {t}')
    #         print('-------------------')
    #         print(f'mat{mat}')
    #         print('-------------------')
    # print(f'mat done{mat.shape}')
    # print(f'mat done{mat}')
    return t

    #print('---------------ny matrise---------------')
    #print(mat)

    #return mat


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
    #fitting the image in the middle, and now I got padding
    out_im[pad_size:-pad_size,pad_size:-pad_size,:] = im
    print(out_im.shape)
    print('----------------------')
  
    kernel_flip = np.flip(kernel)
    
    kernel_new_size = np.tile(kernel_flip[:,:,None],(1,1,image_depth))
    
    
    #print(np.flip(kernel_new_size).shape)
    #im_pad = pad(im,image_height,image_width)
    for y in range(pad_size,image_height_pad-pad_size):
        for x in range(pad_size,image_width_pad-pad_size):
            #print(f'image = {im_conv[x,y,:]}')
            #print(f'image = {im_conv[x,y,:].shape}')
            im_conv[x,y,:] = convole2D_kxk(out_im[x-pad_size:x+pad_size+1,y-pad_size:y+pad_size+1,:],kernel_new_size)
            #break
        #break
            
          
    im = im_conv[pad_size:-pad_size,pad_size:-pad_size,:]
   
    #plt.imshow(out_im[pad_size:-pad_size,pad_size:-pad_size,:])
    #plt.imshow(im)
    #plt.show()
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
    save_im(output_dir.joinpath("im_smoothed_test.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel_test.jpg"), im_sobel)

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
print("--- %s seconds ---" % (time.time() - start_time))