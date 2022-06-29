import cv2
import numpy as np
import torch
from torchvision.models import detection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = detection.retinanet_resnet50_fpn(
    pretrained=True, progress=False, num_classes=91, pretrained_backbone=True
).to(DEVICE)

model.eval()


def crop_human(image, is_rgb=False):
    box = get_human_box(image, is_rgb=is_rgb)
    if box is not None:
        return crop_box(image, box)
    else:
        return None


def crop_box(image, box):
    (start_x, start_y, end_x, end_y) = box.astype("int")
    image = image[start_y:end_y, start_x:end_x]
    return image


def get_human_box(image, is_rgb=False):
    if not is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to(DEVICE)

    detections = model(image)[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]

        if confidence > 0.7:
            idx = int(detections["labels"][i])

            # if human is detected
            if idx == 1:
                box = detections["boxes"][i].detach().cpu().numpy()
                return box.astype("int")

    return None
