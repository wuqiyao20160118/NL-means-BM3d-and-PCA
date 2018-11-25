import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets


def trans_gray_PIL(img_dir):
    # using PIL.Image.convert()
    img = Image.open(img_dir)
    gray_convert = img.convert('L')
    gray_convert.save('gray_starry_night.png')
    gray = np.asarray(gray_convert)
    return gray


def PCA(data, mean, th=0.4, reconstruct=True):
    image = data - mean
    eig_vecs, eig_vals, vh = np.linalg.svd(image)
    sum_eig = (np.cumsum(eig_vals) / np.sum(eig_vals))
    for i in range(sum_eig.shape[0]):
        if sum_eig[i] > th:
            index = i
            break
    transform_mat = eig_vecs[:, :index]
    transform_val = np.zeros((index, index))
    transform_vh = vh.T[:index, :]
    for i in range(index):
        transform_val[i, i] = eig_vals[i]
    score = np.matmul(transform_mat.T, image)
    if reconstruct:
        image = np.matmul(transform_mat, score) + mean
    return image, mean, score


def eigenface_PCA():
    faces = datasets.fetch_olivetti_faces()
    mean = np.mean(faces.data, axis=0)
    diff, avgImg, cov_vector = PCA(faces.data, mean=mean, th=0.8, reconstruct=False)
    plt.figure(3)
    avgImg = np.reshape(avgImg, (64, 64))
    cov_vector = np.reshape(cov_vector, (cov_vector.shape[0], 64, 64))
    plt.imshow(avgImg, cmap="gray")
    fig = plt.figure(figsize=(8, 8))
    # plot several images
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(cov_vector[i], cmap=plt.cm.bone)
        #ax.imshow(faces.images[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == "__main__":
    Img = trans_gray_PIL("./starry_night.jpg")
    plt.figure(1)
    plt.imshow(Img, cmap="gray")
    transformed_img, _, _ = PCA(Img, np.mean(Img))
    plt.figure(2)
    plt.imshow(transformed_img, cmap="gray")
    plt.show()
    eigenface_PCA()

