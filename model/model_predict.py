import numpy as np
import os
import face_recognition
from PIL import Image
import pickle
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class PredictModelImgLR:
    """
    Предсказание на модели логистической регрессии по тестовому изображению

    Args:
        path_load (str): путь до каталога с тестовым изображением
        size_new (int): необходимый размер изображения по одной из сторон
    """

    def __init__(self, path_load: str, size_new: int):
        self.path_load = path_load
        self.size_new = size_new

        # загрузка модели, словаря имён:таргетов
        self.model, self.name_labels = self.__load_data()

    def predict_model(self) -> None:
        """ Предсказание на модели логистической регрессии """

        test_photo_resized_conv = self.__load_image()

        test_face_boxes = face_recognition.face_locations(test_photo_resized_conv)
        # если найдено больше 1 лица на изображении - оно исключается
        if len(test_face_boxes) == 1:
            test_face_encod = face_recognition.face_encodings(test_photo_resized_conv)[0]
            test_predict = self.model.predict([test_face_encod])
            test_predict_name = list(self.name_labels.keys())[list(self.name_labels.values()).index(test_predict)]
            print('predict: %d' % test_predict)
            print('predict name: %s' % test_predict_name)

            test_predict_proba = self.model.predict_proba([test_face_encod])[0][test_predict][0]
            print(test_predict_proba)

    def __load_data(self) -> tuple[np.array, dict]:
        """ Загрузка данных для обучения """

        try:
            path_model = os.path.join(self.path_load, 'model_LR.pkl')
            with open(path_model, 'rb') as file:
                load_model = pickle.load(file)

            path_act = os.path.join(self.path_load, 'name_labels.json')
            with open(path_act, 'r') as file:
                load_name_labels = json.load(file)
        except Exception as ex:
            print(f'Error: {ex}')
        else:
            return load_model, load_name_labels

    def __load_image(self) -> np.array:
        """ Загрузка изображения, с изменением размера """

        path_test_image = os.path.join(self.path_load, 'test_image4.jpg')
        # изменение формата тестового изображения
        with Image.open(path_test_image) as photo:
            test_photo_resized = resize_photo(photo, self.size_new)
            test_photo_resized_conv = np.array(test_photo_resized.convert('RGB'))

        return test_photo_resized_conv


''' Вызов изменения размера изображения из preproicessing!! '''