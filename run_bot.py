import argparse
import os

import telebot
from moviepy.editor import VideoFileClip

from face.face_model import Model as FaceModel
from pose.pose_model import ArousalModel
from service.drawer import Drawer
from service.image_loader import ImageTransformer
from voice.voice_model import VoiceModel

if __name__ == "__main__":
    # Get telegram bot token from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-key", type=str, default="some_token", help="Telegram bot secret token"
    )
    args = parser.parse_args()

    # init all modules
    bot = telebot.TeleBot(args.key, parse_mode=None)
    image_loader = ImageTransformer()
    face_model = FaceModel()
    voice_model = VoiceModel()
    pose_model = ArousalModel()
    drawer = Drawer()

    @bot.message_handler(content_types=["text"])
    def handle_start(message):  # start message reaction
        if message.text == "/start":
            bot.send_message(message.from_user.id, "Send me video or photo")
            bot.register_next_step_handler(message, handle_dock_photo)
        else:
            bot.send_message(
                message.from_user.id,
                "Write /start if you want to use bot",
            )

    @bot.message_handler(content_types=["photo"])
    def handle_dock_photo(message):  # image massage reaction
        try:
            file_info = bot.get_file(message.photo[-1].file_id)
            download_file = bot.download_file(file_info.file_path)

            image = image_loader.image_as_array(download_file)
            prediction, box = face_model.predict(image)

            # Send message

            if prediction is not None:
                text = "This person looks like " + prediction
            else:
                text = "I can't find any face, sorry"
            bot.send_message(message.chat.id, text)

            # Send image
            image = drawer.draw_face_box(image, box)
            image = drawer.draw_face_predict(image, prediction)
            image = image_loader.image_as_bytes(image)
            bot.send_photo(message.chat.id, image)

        except Exception as e:
            bot.reply_to(message, "Can't load image")

    @bot.message_handler(content_types=["video"])
    def handle_dock_video(message):
        try:
            file_info = bot.get_file(message.video.file_id)
            download_file = bot.download_file(file_info.file_path)

            bot.reply_to(message, "Processing")

            input_file_name = "input_video" + str(message.chat.id) + ".mp4"
            output_file_name = "output_video" + str(message.chat.id) + ".mp4"

            if os.path.exists(output_file_name):
                os.remove(output_file_name)

            if os.path.exists(input_file_name):
                os.remove(input_file_name)

            with open(input_file_name, "wb") as input_file:
                input_file.write(download_file)
            voice_model_predict = voice_model.predict(input_file_name)
            video = VideoFileClip(input_file_name)

            # set models & foice predict as dreawer members for frame processing fun
            drawer.set_models(pose_model, face_model, voice_model_predict)

            # input video processing
            newclip = video.fl(drawer.frame_processor, apply_to="mask")
            # save
            newclip.write_videofile(output_file_name)

            with open(output_file_name, "rb") as output_file:
                send_file = output_file.read()

            bot.send_video(message.chat.id, send_file)
            os.remove(output_file_name)
            os.remove(input_file_name)

        except Exception as e:
            bot.reply_to(message, "Something went wrong")

    bot.polling(non_stop=True, interval=0, timeout=1000)
