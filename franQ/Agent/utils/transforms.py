"""
A set of transformations that operate on batches
"""
import numpy as np
from torchvision.transforms import RandomChoice


def ImgToUint8(img_batch: np.ndarray):
    if img_batch.dtype == np.uint8:
        return img_batch
    else:
        # Image is some kind of float in range [0,1]
        assert (img_batch.max() <= 1).all()
        assert (img_batch.min() >= 0).all()
        return (img_batch * 255).astype(np.uint8)


def ImgToFloat(img_batch: np.ndarray):
    if not (img_batch.dtype == np.uint8):
        return img_batch  # already a float
    else:
        assert (img_batch.max() <= 255).all()
        assert (img_batch.min() >= 0).all()
        return img_batch.astype(np.float32) / 255


def random_color_cutout(imgs: np.ndarray, min_cut=0.1, max_cut=0.3):
    """
        args:
        min / max cut: int, min / max size of cutout
        returns np.array
    """

    t, n, c, h, w = imgs.shape
    top = np.random.uniform((max_cut * h, max_cut * w), (h, w)).astype(np.int)
    bottom = top - np.random.uniform((0, 0), (max_cut * h, max_cut * w)).astype(np.int)

    cutouts = imgs.copy()
    cutouts[:, :, :, bottom[0]:top[0], bottom[1]:top[1]] = np.random.uniform(0.0, imgs.max())
    return cutouts


def random_channel_cutout(imgs: np.ndarray, max_cut=0.9):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout
        returns np.array
    """

    t, n, c, h, w = imgs.shape
    top = np.random.uniform((max_cut * h, max_cut * w), (h, w)).astype(np.int)
    bottom = top - np.random.uniform((0, 0), (max_cut * h, max_cut * w)).astype(np.int)
    _min, _max = _channelwise_stats(imgs)
    _max.shape = -1
    _min.shape = -1
    cutouts = imgs.copy()
    dropped_channel = np.random.randint(0, c)
    cutouts[:, :, dropped_channel,
    bottom[0]:top[0], bottom[1]:top[1]] = np.random.uniform(0.0, imgs.max())  # ((_max - _min) / 2)[dropped_channel]
    return cutouts


def rgb2greyscale(imgs: np.ndarray):
    # imgs: b x c x h x w
    orig_type = imgs.dtype
    t, b, c, h, w = imgs.shape
    if c != 3:
        return imgs
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114
    imgs = imgs[:, :, None, :, :]
    imgs = np.repeat(imgs, 3, 2)
    return imgs.astype(orig_type)


def hsv2greyscale(imgs: np.ndarray):
    # imgs: b x c x h x w
    orig_type = imgs.dtype
    t, b, c, h, w = imgs.shape
    if c != 3:
        return imgs
    imgs[:, :, :-1] = np.random.uniform(0, imgs.max())
    return imgs.astype(orig_type)


class Dropout:
    def __init__(self, drop_probability=0.3):
        self.drop_probability = drop_probability

    def __call__(self, imgs):
        return dropout(imgs, self.drop_probability)


def dropout(imgs: np.ndarray, drop_probability=0.3):
    mask = np.random.uniform(size=imgs.shape)
    dtype = imgs.dtype
    _mean, _max = _channelwise_stats(imgs)
    imgs = (mask >= drop_probability) * imgs + \
           (
                   (mask < drop_probability) *
                   np.random.uniform() * _max
               # ((_max - _mean) / 2)
           )
    return imgs.astype(dtype)


def _channelwise_stats(imgs):
    t, n, c, h, w = imgs.shape
    _min, _max = [], []
    for i in range(c):
        _min.append(imgs[:, :, i].min())
        _max.append(imgs[:, :, i].max())
    _min, _max = np.asarray(_min), np.asarray(_max)
    _min.shape = (1, 1, -1, 1, 1)
    _max.shape = (1, 1, -1, 1, 1)
    return _min, _max


def channelwise_unit_variance(imgs: np.ndarray):
    _min, _max = _channelwise_stats(imgs)
    imgs -= _min
    imgs /= (_max - _min)
    return imgs


def random_gamma(imgs: np.ndarray, min=0.5, max=2):
    dtype = imgs.dtype
    imgs = imgs.astype(np.float)
    _min, _max = _channelwise_stats(imgs)
    imgs = channelwise_unit_variance(imgs)
    imgs = imgs ** np.random.uniform(min, max)
    imgs = np.clip(imgs, 0, 1)
    return (imgs * _max).astype(dtype)


def random_gamma_channelwise(imgs: np.ndarray, min=0.5, max=2):
    dtype = imgs.dtype
    imgs = imgs.astype(np.float)
    _min, _max = _channelwise_stats(imgs)
    imgs = channelwise_unit_variance(imgs)
    gamma = np.random.uniform(min, max, (1, 1, 3, 1, 1))
    imgs **= gamma
    return (imgs * _max).astype(dtype)


def random_brightness(imgs: np.ndarray, min=0.5, max=2):
    dtype = imgs.dtype
    prev_max = imgs.max()
    imgs = imgs.astype(np.float)
    offset = np.random.uniform(min, max) * prev_max
    imgs = ((imgs - imgs.min()) / (offset - imgs.min()))  # normalize
    imgs = np.clip(imgs, 0, 1)
    return (imgs * prev_max).astype(dtype)


def random_brightness_channelwise(imgs: np.ndarray, min=0.5, max=1.5):
    dtype = imgs.dtype
    imgs = imgs.astype(np.float)
    _min, _max = _channelwise_stats(imgs)
    imgs = channelwise_unit_variance(imgs)

    scaler = np.random.uniform(min, max, (1, 1, 3, 1, 1))
    imgs *= scaler
    imgs = np.clip(imgs, 0, 1)
    return (imgs * _max).astype(dtype)


def identity(imgs):
    return imgs


channelwise_transforms = RandomChoice([
    random_brightness_channelwise,
    random_gamma_channelwise,
    hsv2greyscale,
    random_channel_cutout,
    dropout,
    identity,
])

if __name__ == '__main__':
    def sanity_check():
        import cv2
        # test_file = r"C:\Users\iHexx\Downloads\1_K6J46bRKr_HC-1bCqlEKYA.jpeg"
        test_file = r"C:\Users\iHexx\Downloads\lena.jpg"
        img: np.ndarray = cv2.imread(test_file)

        cv2.imshow("lenaa", img)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        batch = np.expand_dims(np.moveaxis(img, -1, 0), 0)
        print(batch.shape)

        for i in range(5):
            g = hsv2greyscale((
                batch)
            )

            cv2.imshow(f"graay{i}",
                       cv2.cvtColor(
                           np.moveaxis(
                               g.squeeze(),
                               0, -1),
                           cv2.COLOR_HSV2BGR)
                       )

        cv2.waitKey(0)


    sanity_check()
