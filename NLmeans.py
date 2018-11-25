import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

np.random.seed(1)


def trans_gray_PIL(img_dir):
    # using PIL.Image.convert()
    img = Image.open(img_dir)
    img = img.resize((256, 256))
    gray_convert = img.convert('L')
    gray_convert.save('lena_gray_resize.png')
    gray = np.asarray(gray_convert)
    return gray


def NLmeans(img, kernel_cov=5.0):
    width, height = img.shape
    filter_size = 3  # the radio of the filter
    search_size = 10  # the ratio of the search size
    pad_img = np.pad(img, ((filter_size, filter_size), (filter_size, filter_size)), 'symmetric')
    result = np.zeros(img.shape)
    kernel = np.ones((2 * filter_size + 1, 2 * filter_size + 1))
    kernel = kernel / ((2 * filter_size + 1) ** 2)
    for w in range(width):
        for h in range(height):
            w1 = w + filter_size
            h1 = h + filter_size
            x_pixels = pad_img[w1-filter_size:w1+filter_size+1, h1-filter_size:h1+filter_size+1]
            # x_pixels = np.reshape(x_pixels, (49, 1)).squeeze()
            w_min = max(w1-search_size, filter_size)
            w_max = min(w1+search_size, width+filter_size-1)
            h_min = max(h1-search_size, filter_size)
            h_max = min(h1+search_size, height+filter_size-1)
            sum_similarity = 0
            sum_pixel = 0
            weight_max = 0
            for x in range(w_min, w_max+1):
                for y in range(h_min, h_max+1):
                    if (x == w1) and (y == h1):
                        continue
                    y_pixels = pad_img[x-filter_size:x+filter_size+1, y-filter_size:y+filter_size+1]
                    #print(y_pixels.shape)
                    # y_pixels = np.reshape(y_pixels, (49, 1)).squeeze()
                    distance = x_pixels - y_pixels
                    distance = np.sum(np.multiply(kernel, np.square(distance)))
                    similarity = np.exp(-distance/(kernel_cov*kernel_cov))
                    if similarity > weight_max:
                        weight_max = similarity
                    sum_similarity += similarity
                    sum_pixel += similarity * pad_img[x, y]
            sum_pixel += weight_max * pad_img[w1, h1]
            sum_similarity += weight_max
            if sum_similarity > 0:
                result[w, h] = sum_pixel / sum_similarity
            else:
                result[w, h] = img[w, h]
    return result


def addGaussianNoise(img, percetage):
    num = int(percetage*img.shape[0]*img.shape[1])
    result = img - 0  # if do not add "-0", it will be a read-only numpy ndarray
    for i in range(num):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        result[temp_x][temp_y] = 255
    return result


def SaltAndPepper(img, percetage):
    noiseImg = img - 0
    num = int(percetage*img.shape[0]*img.shape[1])
    for i in range(num):
        x = np.random.randint(0, img.shape[0]-1)

        y = np.random.randint(0, img.shape[1]-1)

        if np.random.randint(0, 2) == 0:
            noiseImg[x, y] = 0
        else:
            noiseImg[x, y] = 255
    return noiseImg


if __name__ == "__main__":
    Img = trans_gray_PIL("./lena.png")
    noised_img = SaltAndPepper(Img, 0.1)
    plt.figure(1)
    plt.imshow(noised_img, cmap="gray")
    #plt.show()
    #denoised_img = NLmeans(noised_img, 3)
    plt.figure(2)
    #plt.imshow(denoised_img, cmap="gray")
    #plt.show()
    #plt.figure(3)
    denoised_img2 = cv2.fastNlMeansDenoising(noised_img, h=20)
    plt.imshow(denoised_img2, cmap="gray")
    plt.show()

