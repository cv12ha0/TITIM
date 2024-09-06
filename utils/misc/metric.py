import math
import numpy as np
from PIL import Image
import cv2


def entropy(image):
    image = Image.fromarray(image)
    image_gray = image.convert('L')
    histogram = image_gray.histogram()
    total_pixels = sum(histogram)
    probs = [float(h) / total_pixels for h in histogram]

    entropy = -sum([p * math.log2(p) for p in probs if p != 0])
    return entropy


def per_pixel(img1, img2):
    # img = cv2.imread('test01.png')
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # to binary
    _, threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    similarity = np.mean(threshold)
    return similarity


def ssim(img1, img2, use_var=False):
    from skimage.metrics import structural_similarity as compare_ssim
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # cal ssim
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    similarity = compare_ssim(img1_gray, img2_gray)

    if use_var:
        img1_var, img2_var = np.var(img1_gray/255), np.var(img2_gray/255)
        var_diff = np.abs(img1_var - img2_var)
        similarity *= (1 - var_diff)
    return similarity


def psnr(img1, img2, data_range=255):
    mse = np.mean((img1/data_range - img2/data_range) ** 2, dtype=np.float64) + 1e-6
    return 20 * math.log10(1 / math.sqrt(mse))


def psnr_depre(img1, img2):
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # cal psnr
    similarity = compare_psnr(img1.astype(float), img2.astype(float), data_range=255)
    return similarity


def cmp_lpips(img1, img2, net='alex'):
    if img1.shape[-1] > 3 or img2.shape[-1] > 3:
        return -1
    if not hasattr(cmp_lpips, 'loss_fn'):
        import lpips
        cmp_lpips.loss_fn = lpips.LPIPS(net=net)  # alex vgg
        cmp_lpips.im2tensor = lpips.im2tensor
    # similarity = cmp_lpips.loss_fn(img1, img2)
    # similarity = cmp_lpips.loss_fn(torch.tensor(img1.transpose((2, 0, 1))), torch.tensor(img1.transpose((2, 0, 1)))).detach().numpy().squeeze()
    similarity = cmp_lpips.loss_fn(cmp_lpips.im2tensor(img1), cmp_lpips.im2tensor(img2)).detach().numpy().squeeze()
    return float(similarity)


def hist(img1, img2):
    # to HSV
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    # cal histogram
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX, -1)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX, -1)

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity



def harris(img1, img2):
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    dest1 = cv2.cornerHarris(img1, 2, 5, 0.07)
    dest2 = cv2.cornerHarris(img2, 2, 5, 0.07)
    # dest = cv2.dilate(dest1, None) 
    # image[dest > 0.01 * dest.max()] = [0, 0, 255] 
    pass


def sift(img1, img2):
    # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create SIFT feature extractor 
    sift = cv2.SIFT_create()
    # detect key points & compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    if descriptors1 is None or descriptors2 is None:
        return 0

    # create FLANN feature matcher
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # filter & remain good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))
    return similarity


def surf(img1, img2):
    # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create SURF feature extractor 
    surf = cv2.xfeatures2d_SURF.create()

    # detect key points & compute descriptors
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

    # create FLANN feature matcher
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # filter & remain good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))
    return similarity


def orb(img1, img2):
    # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create SURF feature extractor 
    orb = cv2.ORB_create()

    # detect key points & compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    if descriptors1 is None or descriptors2 is None:
        return 0

    # create BFMatcher (feature matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # filter & remain good matches
    good_matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches))]  # * 0.15

    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))
    return similarity


if __name__ == '__main__':
    img1 = cv2.imread('image01.jpg')
    img2 = cv2.imread('image01_r.png')
    img3 = cv2.resize(cv2.imread('image02.png'), (img1.shape[1], img1.shape[0]))
