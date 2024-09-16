import cv2
import numpy as np
import random

def sp_noise(img,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(img.shape,np.uint8)
    rdn = random.random()
    thres = 1 - prob 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

def sepia_trans(img):
    '''
    Modify the image to sepia by converting the image to grayscale
    to capture information about the intensity of the image
    add the characteristic brownish tone to the image
    then modulate the image with the originals intensity
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32)/255
    sepia = np.ones(img.shape)
    sepia[:,:,0] *= 153 #B
    sepia[:,:,1] *= 204 #G
    sepia[:,:,2] *= 255 #R
    sepia[:,:,0] *= normalized_gray #B
    sepia[:,:,1] *= normalized_gray #G
    sepia[:,:,2] *= normalized_gray #R
    return np.array(sepia, np.uint8)

def make_border(img,h_scale = 0.04,v_scale = 0.02):
    '''
    Add border to the image
    '''
    horizontal = int(h_scale * img.shape[0])
    vertical = int(v_scale * img.shape[1])
    return cv2.copyMakeBorder(img, vertical, vertical, horizontal, horizontal, cv2.BORDER_CONSTANT, value=(62, 63, 64))

def create_vignette(img, level=2):
    '''
    Create a vignette effect on the image
    '''
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/level)
    kernel_y = cv2.getGaussianKernel(rows, rows/level)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)
    vignette = np.copy(img)
    for i in range(3):
        vignette[:,:,i] = vignette[:,:,i] * mask
    return vignette

def change_contrast(image, alpha = 0.7, beta=1):
    '''
    Change the contrast of the image with alpha and brightness with beta
    '''
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_vintage_effect(image_path, output_path):
    '''
    Main function for handling the transformation
        It reads the image 
        Dilates the image - Add noise [X]
        Uses Gaussian blur to make the image blurry - Add noise [X]
        Adds salt and pepper noise to the image - Add noise [X]
        Converts the image to sepia - Add image color transformation [X]
        Creates a vignette effect - Add vignette effect [X]
        Adds a border to the image - Add retro border [X]
        Lower the contrast of the image - Change contrast [X]

    These complete the requirements for the homework
        Image color transformation [X] - with sepia filter
        Add vignette effect [X] - with vignette effect
        Add retro border [X] - with border
        Low contrast [X] - with contrast change
        Noises and patches [X] - with salt and pepper noise, gaussian blur, dilation
    '''
    # image reading
    img = cv2.imread(image_path)

    # image dilation or erosion, haven't decided yet
    kernel = np.ones((7,7), np.uint8)
    dilate_erode = cv2.dilate(img, kernel)
    
    # gaussian blur to make image blurry
    blurred = cv2.GaussianBlur(dilate_erode, (7, 7),0)
    noisyi = sp_noise(blurred, 0.05)
    
    sepia = sepia_trans(noisyi)

    # create vignette effect
    vignette = create_vignette(sepia, level=4)

    # add border to the image
    borders = make_border(vignette, h_scale = 0.04, v_scale = 0.02)

    # change contrast
    oldified = change_contrast(borders, alpha=0.7, beta=1)

    cv2.imwrite(output_path, oldified)

    cv2.imshow('original picture',img)
    cv2.imshow('oldified picture',borders)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"The transformation is complete and saved to: {output_path}")

if __name__ == "__main__":
    input_image = "./instakep.jpeg"  # Cserélje ki a saját képének elérési útjával
    output_image = "./vintage_output.jpg"
    apply_vintage_effect(input_image, output_image)