import glob
import shutil
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io
import torch
import torch.utils.data
import torchvision.transforms as transforms
from moviepy.editor import *
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix


def get_transforms(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
    )
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


def overall_accuracy(validation_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network: %d %%" % (100 * correct / total))


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def get_predct_onnx(ort_session, validation_loader, classes):

    full_predicted = []
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
            ort_outs = ort_session.run(None, ort_inputs)
            prediction = ort_outs[0]

            prediction = np.argmax(prediction)
            full_predicted.append(classes[prediction])

    c = Counter(full_predicted)
    stats = dict(c)
    predict = max(stats, key=stats.get)
    return predict


def get_predict(validation_loader, model, classes):
    model.eval()
    full_predicted = []
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for el in predicted:
                full_predicted.append(classes[el])
    c = Counter(full_predicted)
    stats = dict(c)
    predict = max(stats, key=stats.get)
    return predict


def from_video_to_audio(video_name):

    if not os.path.exists("voice/audio_from_video"):
        os.mkdir("voice/audio_from_video")
    if not os.path.exists("voice/img_data"):
        os.mkdir("voice/img_data")
    if not os.path.exists("voice/img_data/img"):
        os.mkdir("voice/img_data/img")
    for file in glob.glob(video_name):
        basename = os.path.basename(file)

        target_path = "voice/audio_from_video/%s.wav" % video_name.split(".")[0]
        video = VideoFileClip(video_name)
        video.audio.write_audiofile(target_path)

        new_audio = AudioSegment.from_file(target_path)
        step = 3 * 1000
        t1 = 0
        t2 = step
        temp_path = target_path.split("/")
        col = int((len(new_audio) / step))

        for i, j in enumerate(range(col)):

            temp_audio = new_audio[t1:t2]
            temp_audio.export(
                f'{temp_path[0]}/{temp_path[1]}/{temp_path[2].replace(".","")}{i}.wav',
                format="wav",
            )
            t1 = t1 + step
            t2 = t2 + step
        new_audio.export(f"{target_path}", format="wav")
        os.remove(target_path)
        target_path = temp_path[0] + "/" + temp_path[1] + "/*.wav"

        base_folder = from_audio_to_image(target_path)
    shutil.rmtree("voice/audio_from_video")
    return base_folder


def from_audio_to_image(path_audio):
    n_mels = 128
    base_folder = "voice/img_data"
    for files in glob.glob(path_audio):
        basename = os.path.basename(files)
        basename = basename.split(".")[0]
        basename = basename.replace("wav", "_")
        x, sr = librosa.load(files, mono=True, sr=22050)
        out = f"voice/img_data/img/{basename}.png"
        spectrogram_image(x, sr=sr, out=out, n_mels=n_mels)
    return base_folder


def metric_for_each_class(validation_loader, model, classes):
    class_correct = list(0.0 for i in range(7))
    class_total = list(0.0 for i in range(7))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data

            y_true.append(labels.cpu().detach().numpy())
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted.cpu().detach().numpy())
            c = (predicted == labels).squeeze(0)

            for i in range(len(labels)):
                label = labels[i]
                if len(labels) > 1:
                    class_correct[label] += c[i].item()
                else:
                    class_correct[label] += c.item()
                    break
                class_total[label] += 1

    for i in range(len(classes)):
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i])
        )
    cf_matrix(y_true, y_pred, classes)


def cf_matrix(y_true, y_pred, classes):
    array_y_pred = []
    for el in y_pred:
        for i in el:
            array_y_pred.append(i)
    array_y_true = []
    for el in y_true:
        for i in el:
            array_y_true.append(i)
    cf_matrix = confusion_matrix(array_y_true, array_y_pred)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues")

    # drawing the plot
    ax.set_title("Seaborn Confusion Matrix with labels\n\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ")

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # Display the visualization of the Confusion Matrix.
    plt.show()
