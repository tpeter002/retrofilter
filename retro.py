import cv2
import numpy as np
import random

def add_sp_noise(img,prob):
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

def add_sepia(img):
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

def add_vignette(img, level=2):
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

def load_texture(file_path, size):
    '''
    Load and prepare a texture image.
    '''
    texture = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(texture, size, interpolation=cv2.INTER_AREA)
    return resized

def apply_texture(image, texture, intensity=0.5):
    '''
    Apply a texture to the image using the specified blend mode.
    '''
    # Ensure texture has the same number of channels as the image
    if len(image.shape) == 3 and image.shape[2] == 3:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    
    # Ensure both images have the same data type
    image = image.astype(np.float32)
    texture = texture.astype(np.float32)
    
    # Overlay the images
    low = 2 * image * texture / 255.0
    high = 255.0 - 2 * (255.0 - image) * (255.0 - texture) / 255.0
    blended = np.where(image < 128, low, high)
    # Apply the blending with intensity
    result = image * (1 - intensity) + blended * intensity
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_vintage_effect(image_path, output_path):
    '''
    Main function for handling the transformation
        It reads the image 
        Dilates the image - Add noise [X]
        Uses Gaussian blur to make the image blurry - Add noise [X]
        Adds salt and pepper noise to the image - Add noise [X]
        Converts the image to sepia - Add image color transformation [X]
        Creates a vignette effect - Add vignette effect [X]
        Adds an old border to the image - Add retro border [X]
        Lower the contrast of the image - Change contrast [X]
        Add a stained picture for old effect
    '''
    # image reading
    img = cv2.imread(image_path)

    # image dilation or erosion, haven't decided yet
    kernel = np.ones((3,3), np.uint8)
    dilate_erode = cv2.dilate(img, kernel)
    
    # gaussian blur to make image blurry
    blurred = cv2.GaussianBlur(dilate_erode, (3, 3),0)
    noisyi = add_sp_noise(blurred, 0.05)
    
    sepia = add_sepia(noisyi)

    # create vignette effect
    vignette = add_vignette(sepia, level=2)

    # add border to the image
    bordered = load_texture("input_images/border_overlay.jpeg", (vignette.shape[1], vignette.shape[0]))
    new_border = apply_texture(vignette, bordered, intensity=1)

    # change contrast
    oldified = change_contrast(new_border, alpha=0.7, beta=1)

    old_pic = load_texture("input_images/old_texture.jpeg", (oldified.shape[1], oldified.shape[0]))
    stained_pic = apply_texture(oldified, old_pic, intensity=1)

    
    cv2.imwrite(output_path, stained_pic)

    cv2.imshow('original picture',img)
    cv2.imshow('oldified picture',stained_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"The transformation is complete and saved to: {output_path}")

if __name__ == "__main__":
    input_image = "./old_times_tp.jpg"  # Cserélje ki a saját képének elérési útjával
    output_image = "./vintage_tp.jpg"
    apply_vintage_effect(input_image, output_image)