import cv2
import numpy as np
import torch
from torchvision import transforms


class Model:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.class_names = {
            0: "Surprise",
            1: "Fear",
            2: "Disgust",
            3: "Happiness",
            4: "Sadness",
            5: "Anger",
            6: "Neutral",
        }
        self.face_bboxer = cv2.dnn.readNetFromCaffe(
            "./saved_models/face/deploy.prototxt",
            "./saved_models/face/res10_300x300_ssd_iter_140000.caffemodel",
        )
        if torch.cuda.is_available():
            self.emotion_classifier = torch.jit.load("./saved_models/face/ResnetRUL_cuda.pth")
            self.device = 'cuda'
        else:
            self.emotion_classifier = torch.jit.load("./saved_models/face/ResnetRUL_cpu.pth")
            self.device = 'cpu'

    def predict(self, image):

        h = image.shape[0]
        w = image.shape[1]

        # default prediction for missing face
        box = None
        str_prediction = None

        # Get face_crop
        self.face_bboxer.setInput(
            cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
            )
        )
        bboxes = self.face_bboxer.forward()

        confidence = bboxes[0, 0, 0, 2]
        if confidence > 0.5:
            box = bboxes[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            if (0 < y < y1 < h) and (0 < x < x1 < w):
                crop = image[y:y1, x:x1]
                topred = self.transform(crop).unsqueeze(0).to(self.device)
                pred = self.emotion_classifier(topred).argmax(axis=-1).item()
                str_prediction = self.class_names[pred]

        return str_prediction, box
