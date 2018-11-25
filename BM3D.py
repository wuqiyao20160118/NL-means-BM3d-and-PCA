import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
import cv2

np.random.seed(1)


# Parameters initialization
sigma = 25

Threshold_Hard3D = 2.7*sigma           # Threshold for Hard Thresholding
First_Match_threshold = 2500             # 用于计算block之间相似度的阈值
Step1_max_matched_cnt = 16              # 组最大匹配的块数
Step1_Blk_Size = 8                     # block_Size即块的大小，8*8
Step1_Blk_Step = 3                      # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 3                   # 块的搜索step
Step1_Search_Window = 39                # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

Second_Match_threshold = 400           # 用于计算block之间相似度的阈值
Step2_max_matched_cnt = 32
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39
Beta_Kaiser = 2.0


def trans_gray_PIL(img_dir):
    # using PIL.Image.convert()
    img = Image.open(img_dir)
    img = img.resize((256, 256))
    gray_convert = img.convert('L')
    gray_convert.save('lena_gray_resize.png')
    gray = np.asarray(gray_convert)
    return gray


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


def Locate_blk(i, j, blk_step, block_size, width, height):
    if i*blk_step+block_size < width:
        point_x = i*blk_step
    else:
        point_x = width - block_size
    if j*blk_step+block_size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_size
    blockPoint = np.array((point_x, point_y), dtype=int)
    return blockPoint


def Define_SearchWindow(img, BlockPoint, WindowSize, Blk_Size):
    """
    :param img: input image
    :param BlockPoint: coordinate of the left-top corner of the block
    :param WindowSize: size of the search window
    :param Blk_Size:
    :return: left-top corner point of the search window
    """
    point_x = BlockPoint[0]
    point_y = BlockPoint[1]
    # get four corner points
    x_min = point_x + Blk_Size/2 - WindowSize/2
    y_min = point_y + Blk_Size/2 - WindowSize/2
    x_max = x_min + WindowSize
    y_max = y_min + WindowSize
    # check whether the corner points have out of range
    if x_min < 0:
        x_min = 0
    elif x_max > img.shape[0]:
        x_min = img.shape[0] - WindowSize
    if y_min < 0:
        y_min = 0
    elif y_max > img.shape[0]:
        y_min = img.shape[0] - WindowSize
    return np.array((x_min, y_min), dtype=int)


def step1_fast_match(img, BlockPoint):
    x, y = BlockPoint
    blk_size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    window_size = Step1_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)
    similar_blocks_3d = np.zeros((max_matched, blk_size, blk_size), dtype=float)
    image = img[x:x+blk_size, y:y+blk_size]
    dct_img = cv2.dct(image.astype(np.float64))
    window_location = Define_SearchWindow(img, BlockPoint, window_size, blk_size)
    blk_num = int((window_size - blk_size) / Search_Step)
    window_x, window_y = window_location

    count = 0
    matched_blk_pos = np.zeros((blk_num**2, 2), dtype=int)
    matched_distance = np.zeros(blk_num**2, dtype=float)
    similar_blocks = np.zeros((blk_num ** 2, blk_size, blk_size), dtype=float)

    for i in range(blk_num):
        for j in range(blk_num):
            search_img = img[window_x:window_x+blk_size, window_y:window_y+blk_size]
            dct_search_img = cv2.dct(search_img.astype(np.float64))
            distance = np.linalg.norm((dct_img - dct_search_img)) ** 2 / (blk_size ** 2)
            if 0 < distance < Threshold:
                matched_blk_pos[count] = np.array((window_x, window_y))
                matched_distance[count] = distance
                similar_blocks[count] = dct_search_img
                count += 1
            window_y += Search_Step
        window_x += Search_Step
        window_y = window_location[1]
    distance = matched_distance[:count]
    sort_index = distance.argsort()

    if count >= max_matched:
        count = max_matched
    else:
        count += 1  # add the template image

    similar_blocks_3d[0] = dct_img
    blk_positions[0] = np.array((x, y))
    for i in range(1, count):
        index = sort_index[i-1]
        similar_blocks_3d[i] = similar_blocks[index]
        blk_positions[i] = matched_blk_pos[index]
    return similar_blocks_3d, blk_positions, count


def step1_3DFiltering(similar_blocks):
    nonzero_count = 0
    for i in range(similar_blocks.shape[1]):
        for j in range(similar_blocks.shape[2]):
            harr_img = cv2.dct(similar_blocks[:, i, j].astype(np.float64))
            harr_img[np.abs(harr_img) < Threshold_Hard3D] = 0
            nonzero_count += harr_img.nonzero()[0].size
            similar_blocks[:, i, j] = cv2.idct(harr_img)[0]
    return similar_blocks, nonzero_count


def integ_hardthreshold(similar_blocks, blk_positions, basic_img, weight_img, nonzero_count, matched_num, Kaiser=None):
    blk_shape = similar_blocks.shape
    if nonzero_count < 1:
        nonzero_count = 1
    block_wight = (1. / nonzero_count) #* Kaiser
    for i in range(matched_num):
        point = blk_positions[i, :]
        temp_img = (1. / nonzero_count) * cv2.idct(similar_blocks[i, :, :]) #* Kaiser
        basic_img[point[0]:point[0] + blk_shape[1], point[1]:point[1] + blk_shape[2]] += temp_img
        weight_img[point[0]:point[0] + blk_shape[1], point[1]:point[1] + blk_shape[2]] += block_wight
    return basic_img, weight_img


def BM3D_step_1(img):
    width, height = img.shape
    block_size = Step1_Blk_Size
    blk_step = Step1_Blk_Step
    width_num = int((width - block_size) / blk_step)
    height_num = int((height - block_size) / blk_step)
    filtered_img = np.zeros(img.shape, dtype=float)
    filter_weight = np.zeros(img.shape, dtype=float)

    #K = np.kaiser(block_size, Beta_Kaiser)
    #m_Kaiser = np.matmul(K.T, K)  # construct a Kaiser window

    for i in range(width_num+2):
        for j in range(height_num+2):
            BlockPoint = Locate_blk(i, j, blk_step, block_size, width, height)
            similar_blocks_3d, blk_positions, count = step1_fast_match(img, BlockPoint)
            similar_blocks, nonzero_count = step1_3DFiltering(similar_blocks_3d)
            filtered_img, filter_weight = integ_hardthreshold(similar_blocks, blk_positions, filtered_img, filter_weight, nonzero_count, count)
    filtered_img /= filter_weight
    filtered_img.astype(np.uint8)
    return filtered_img


def step2_fast_match(basic_img, img, BlockPoint):
    x, y = BlockPoint
    blk_size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    window_size = Step2_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)
    similar_blocks_3d = np.zeros((max_matched, blk_size, blk_size), dtype=float)
    basic_similar_blocks_3d = np.zeros((max_matched, blk_size, blk_size), dtype=float)

    basic_image = basic_img[x:x + blk_size, y:y + blk_size]
    basic_dct_img = cv2.dct(basic_image.astype(np.float64))
    image = img[x:x + blk_size, y:y + blk_size]
    dct_img = cv2.dct(image.astype(np.float64))

    window_location = Define_SearchWindow(img, BlockPoint, window_size, blk_size)
    blk_num = int((window_size - blk_size) / Search_Step)
    window_x, window_y = window_location

    count = 0
    matched_blk_pos = np.zeros((blk_num ** 2, 2), dtype=int)
    matched_distance = np.zeros(blk_num ** 2, dtype=float)
    similar_blocks = np.zeros((blk_num ** 2, blk_size, blk_size), dtype=float)

    for i in range(blk_num):
        for j in range(blk_num):
            search_img = basic_img[window_x:window_x+blk_size, window_y:window_y+blk_size]
            dct_search_img = cv2.dct(search_img.astype(np.float64))
            distance = np.linalg.norm((dct_img - dct_search_img)) ** 2 / (blk_size ** 2)

            if 0 < distance < Threshold:
                matched_blk_pos[count] = np.array((window_x, window_y))
                matched_distance[count] = distance
                similar_blocks[count] = dct_search_img
                count += 1
            window_y += Search_Step
        window_x += Search_Step
        window_y = window_location[1]
    distance = matched_distance[:count]
    sort_index = distance.argsort()

    if count >= max_matched:
        count = max_matched
    else:
        count += 1  # add the template image

    basic_similar_blocks_3d[0] = basic_dct_img
    similar_blocks_3d[0] = dct_img
    blk_positions[0] = np.array((x, y))
    for i in range(1, count):
        index = sort_index[i - 1]
        basic_similar_blocks_3d[i] = similar_blocks[index]
        blk_positions[i] = matched_blk_pos[index]

        x, y = blk_positions[i]
        temp_noisy_img = noised_img[x:x+blk_size, y:y+blk_size]
        similar_blocks_3d[i] = cv2.dct(temp_noisy_img.astype(np.float64))

    return basic_similar_blocks_3d, similar_blocks_3d, blk_positions, count


def step2_3DFiltering(similar_basic_blocks, similar_blocks):
    img_shape = similar_basic_blocks.shape
    Wiener_weight = np.zeros((img_shape[1], img_shape[2]), dtype=float)

    for i in range(img_shape[1]):
        for j in range(img_shape[2]):
            temp_vector = similar_basic_blocks[:, i, j]
            dct_temp = cv2.dct(temp_vector)
            norm2 = np.matmul(dct_temp.T, dct_temp)
            filter_weight = norm2 / (norm2 + sigma**2)
            if filter_weight != 0:
                Wiener_weight[i, j] = 1 / (filter_weight ** 2)
                # Wiener_weight[i, j] = 1 / (filter_weight**2 * sigma**2)
            temp_vector = similar_blocks[:, i, j]
            dct_temp = cv2.dct(temp_vector) * filter_weight
            similar_basic_blocks[:, i, j] = cv2.idct(dct_temp)[0]

    return similar_basic_blocks, Wiener_weight


def integ_Wiener(similar_blocks, Wiener_weight, blk_positions, basic_img, weight_img, matched_num):
    img_shape = similar_blocks.shape
    block_weight = Wiener_weight

    for i in range(matched_num):
        point = blk_positions[i]
        temp_img = block_weight * cv2.idct(similar_blocks[i, :, :])
        basic_img[point[0]:point[0]+img_shape[1], point[1]:point[1]+img_shape[2]] += temp_img
        weight_img[point[0]:point[0]+img_shape[1], point[1]:point[1]+img_shape[2]] += block_weight
    return basic_img, weight_img


def BM3D_step_2(basic_img, img):
    width, height = img.shape
    block_size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    width_num = int((width - block_size) / blk_step)
    height_num = int((height - block_size) / blk_step)
    filtered_img = np.zeros(img.shape, dtype=float)
    filter_weight = np.zeros(img.shape, dtype=float)

    for i in range(width_num+2):
        for j in range(height_num+2):
            BlockPoint = Locate_blk(i, j, blk_step, block_size, width, height)
            basic_similar_blocks_3d, similar_blocks_3d, blk_positions, count = step2_fast_match(basic_img, img, BlockPoint)
            similar_basic_blocks, Wiener_weight = step2_3DFiltering(basic_similar_blocks_3d, similar_blocks_3d)
            filtered_img, filter_weight = integ_Wiener(similar_basic_blocks, Wiener_weight, blk_positions, filtered_img,
                                                              filter_weight, count)
    filtered_img /= filter_weight
    filtered_img.astype(np.uint8)

    return filtered_img


if __name__ == "__main__":
    Img = trans_gray_PIL("./lena.png")
    noised_img = SaltAndPepper(Img, 0.1)
    denoised_img = BM3D_step_1(noised_img)
    final_denoised_img = BM3D_step_2(denoised_img, noised_img)
    plt.figure(1)
    plt.imshow(denoised_img, cmap="gray")
    plt.figure(2)
    plt.imshow(final_denoised_img, cmap="gray")
    plt.figure(3)
    plt.imshow(noised_img, cmap="gray")
    plt.show()


