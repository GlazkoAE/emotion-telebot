import numpy as np
import onnxruntime as ort
import cv2
from keras.applications.inception_v3 import preprocess_input

from pose.detector.person_detector import crop_human


class ArousalModel:
    def __init__(self,
                 seq_length=30,
                 arousal_model_path="./saved_models/pose/arousal_model.onnx",
                 feature_extractor_path="./saved_models/pose/feature_extractor.onnx"):

        # Init both models
        self.feature_extractor_sess = ort.InferenceSession(feature_extractor_path, providers=["CUDAExecutionProvider"])
        self.arousal_model_sess = ort.InferenceSession(arousal_model_path, providers=["CUDAExecutionProvider"])

        # Get input image shape
        self.img_shape = self.feature_extractor_sess.get_inputs()[0].shape[1:3]

        # Init LSTM inputs as zeros
        self.features = np.zeros((1, seq_length, 2048))

    def extract_features(self, img):
        img = cv2.resize(img, self.img_shape)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = np.array(self.feature_extractor_sess.run(["avg_pool"], {"input_img": img}))
        features = np.reshape(features, (1, 1, features.shape[2]))
        return features

    def predict(self, image):
        human = crop_human(image, is_rgb=True)
        if human is None:
            return None

        features = self.extract_features(human)
        self.features = np.concatenate((self.features[:, 1:, :], features), axis=1)

        predict = self.arousal_model_sess.run(["dense_1"], {"input_features": self.features})

        return predict[0][0]
