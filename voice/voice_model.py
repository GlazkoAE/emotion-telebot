import shutil

import onnxruntime as ort
import torch.utils.data
import torchvision

from voice import utils


class VoiceModel:
    def __init__(
        self,
    ):
        # load from onnx
        model_path = "./saved_models/voice/resnet50_voice_all_dataset.onnx"
        img_size = 64

        self.ort_session = ort.InferenceSession(model_path)
        self.transform = utils.get_transforms(img_size)

    def predict(self, video_name):

        classes = (
            "angry",
            "disgust",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprised",
        )

        path_to_data = utils.from_video_to_audio(video_name)

        valid_set = torchvision.datasets.ImageFolder(
            root=path_to_data, transform=self.transform
        )
        validation_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=1, shuffle=False, num_workers=2
        )

        predicted = utils.get_predict_onnx(self.ort_session, validation_loader, classes)

        shutil.rmtree(path_to_data)
        return predicted
