import colorsys
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def display_images(images, titles=None, cols=3, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def color_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    color_map = random_colors(num_classes, bright=False)
    colored_mask = np.zeros((*mask.shape[:-1], 3))

    for clss in range(1, num_classes):
        clss_mask = (mask == clss) * color_map[clss] * 255
        colored_mask += clss_mask
        # [].append(clss_mask)

    return colored_mask



if __name__ == '__main__':
    print(random_colors(10))
    img = np.asarray(Image.open('./samples/5.jpg'))
    display_images([img,img,img])