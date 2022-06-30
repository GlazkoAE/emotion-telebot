import cv2


class Drawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 1
        self.text_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.thickness = cv2.FILLED
        self.margin = 10
        self.face_pos = (20, 50)

        self.text_height = cv2.getTextSize(
            "qwerty42", self.font, self.scale, self.thickness
        )[0][1]
        self.voice_pos = self.face_pos[:1] + (
            self.face_pos[1] + self.text_height + self.margin,
        )
        self.pose_pos = self.voice_pos[:1] + (
            self.voice_pos[1] + self.text_height + self.margin,
        )

        self.models = {"pose_model": None, "face_model": None, "voice_prediction": None}

    def set_models(self, pose_model, face_model, voice_prediction):
        self.models["pose_model"] = pose_model
        self.models["face_model"] = face_model
        self.models["voice_prediction"] = voice_prediction

    def frame_processor(self, get_frame, t):
        """frame processor for moviepy.VideoFileClip.fl()
        https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.VideoClip.fl
        """
        frame = get_frame(t)

        if self.models["pose_model"] is not None:
            pose_prediction = self.models["pose_model"].predict(frame)
        else:
            pose_prediction = 'None'

        if self.models["face_model"] is not None:
            face_prediction, box = self.models["face_model"].predict(frame)
        else:
            face_prediction = 'None'
            box = None

        if self.models["voice_prediction"] is not None:
            voice_prediction = self.models["voice_prediction"]
            self.models["voice_prediction"] = self.models["voice_prediction"][1:]
        else:
            voice_prediction = 'None'

        frame = self.draw_face_box(frame, box)
        frame = self.draw_face_predict(frame, face_prediction)
        frame = self.draw_pose_predict(frame, pose_prediction)
        frame = self.draw_voice_predict(frame, voice_prediction)

        return frame

    def draw_face_predict(self, image, predict):
        text = (
            "Face was not founded"
            if predict is None
            else "Facial expression: " + predict
        )
        rectangle_pos = self._get_rectangle_pos(self.face_pos, text)
        cv2.rectangle(
            image, self.face_pos, rectangle_pos, self.bg_color, self.thickness
        )
        cv2.putText(
            image,
            text,
            self.face_pos,
            self.font,
            self.scale,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

        return image

    def draw_pose_predict(self, image, predict):
        text = (
            "Person was not founded" if predict is None else "Arousal: " + str(predict)[:6]
        )
        rectangle_pos = self._get_rectangle_pos(self.pose_pos, text)
        cv2.rectangle(
            image, self.pose_pos, rectangle_pos, self.bg_color, self.thickness
        )
        cv2.putText(
            image,
            text,
            self.pose_pos,
            self.font,
            self.scale,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

        return image

    def draw_voice_predict(self, image, predict):
        text = "Voice intonation: " + predict
        rectangle_pos = self._get_rectangle_pos(self.voice_pos, text)
        cv2.rectangle(
            image, self.voice_pos, rectangle_pos, self.bg_color, self.thickness
        )
        cv2.putText(
            image,
            text,
            self.voice_pos,
            self.font,
            self.scale,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

        return image

    def _get_rectangle_pos(self, pos, text):
        txt_size = cv2.getTextSize(text, self.font, self.scale, self.thickness)
        end_x = pos[0] + txt_size[0][0] + self.margin
        end_y = pos[1] - txt_size[0][1] - self.margin
        return end_x, end_y

    @staticmethod
    def draw_face_box(image, box):
        if box is not None:
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
        return image
