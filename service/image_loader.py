from io import BytesIO

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class ImageTransformer:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.resize = cv2.resize

    @staticmethod
    def image_as_array(image_bytes: bytes):
        image = np.array(Image.open(BytesIO(image_bytes)))
        return image

    @staticmethod
    def image_as_bytes(image: np.array):
        image = Image.fromarray(image)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return img_byte_arr.getvalue()
