import numpy as np
import os
import face_recognition
from PIL import Image
import pickle
import json
import yaml
import preprocessing

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

PATH_LOAD_MODEL = os.path.abspath(os.path.join('..', *config['model']['path']))
SIZE_USERS_PHOTO_NEW = config['predict']['size_image_users']


class PredictModelImgLR:
    """
    Предсказание на модели логистической регрессии по тестовому изображению

    Args:
        path_load_image (str): путь до тестового изображения
        size_new (int): необходимый размер изображения по одной из сторон
        path_load_model (str): путь до тестового изображения
    """

    def __init__(self, path_load_image: str, size_new: int = 512, path_load_model: str = PATH_LOAD_MODEL):
        self.path_load_image = path_load_image
        self.size_new = size_new
        self.path_load_model = path_load_model

        # загрузка модели, словаря имён:таргетов
        self.model, self.name_labels = self.__load_data()

    def predict_model(self):
        """ Предсказание на модели логистической регрессии """

        test_photo_resized_conv = self.__load_image()

        test_face_boxes = face_recognition.face_locations(test_photo_resized_conv)
        # если найдено больше 1 лица на изображении - оно исключается
        if len(test_face_boxes) == 1:
            test_face_encod = face_recognition.face_encodings(test_photo_resized_conv)[0]
            test_predict = self.model.predict([test_face_encod])
            test_predict_name = list(self.name_labels.keys())[list(self.name_labels.values()).index(test_predict)]
            test_predict_proba = self.model.predict_proba([test_face_encod])[0][test_predict][0]

            # print("predict: %d" % test_predict)
            # print("predict name: %s" % test_predict_name)
            # print(test_predict_proba)

            return test_predict, test_predict_name, test_predict_proba


    def __load_data(self) -> tuple[np.array, dict]:
        """ Загрузка данных для обучения """

        try:
            path_model = os.path.join(self.path_load_model, 'model_LR.pkl')
            with open(path_model, 'rb') as file:
                load_model = pickle.load(file)

            path_act = os.path.join(self.path_load_model, 'name_labels.json')
            with open(path_act, 'r') as file:
                load_name_labels = json.load(file)
        except Exception as ex:
            print(f'Error: {ex}')
        else:
            return load_model, load_name_labels

    def __load_image(self) -> np.array:
        """ Загрузка изображения, с изменением размера """

        # path_test_image = os.path.join(self.path_load, 'user_photo.jpg')
        # изменение формата тестового изображения
        with Image.open(self.path_load_image) as photo:
            test_photo_resized = preprocessing.resize_photo(photo, self.size_new)
            test_photo_resized_conv = np.array(test_photo_resized.convert('RGB'))

        return test_photo_resized_conv


# что за хрень
def predict_model_img_lr():
    path_dir = os.path.abspath(os.path.join('..', *config['predict']['path']))
    user_name = 'dovolno_slov'
    user_photo_name = f'{user_name}_photo.jpg'
    path_load = os.path.join(path_dir, user_name, user_photo_name)

    predict_model_img_lr = PredictModelImgLR(path_load, SIZE_USERS_PHOTO_NEW)
    test_predict, test_predict_name, test_predict_proba = predict_model_img_lr.predict_model()

    print("predict: %d" % test_predict)
    print("predict name: %s" % test_predict_name)
    print(test_predict_proba)


if __name__ == "__main__":
    predict_model_img_lr()
