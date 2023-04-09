import os
import face_recognition
from PIL import Image
import pickle
import json
import yaml
import pandas as pd
import numpy as np
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

    def __init__(self, path_load_image: str,
                 size_new: int = SIZE_USERS_PHOTO_NEW, path_load_model: str = PATH_LOAD_MODEL):
        self.path_load_image = path_load_image
        self.size_new = size_new
        self.path_load_model = path_load_model

        # загрузка модели, словаря имён:таргетов
        self.model, self.name_targets = self.__load_data()

    def predict_model(self):
        """ Предсказание на модели логистической регрессии """

        photo_resized_conv = self.__load_image()

        face_boxes = face_recognition.face_locations(photo_resized_conv)
        # если найдено больше 1 лица на изображении - оно исключается
        if len(face_boxes) == 1:
            face_encod = face_recognition.face_encodings(photo_resized_conv)[0]
            pred_target_top = self.model.predict([face_encod])
            pred_name_top = list(self.name_targets.keys())[list(self.name_targets.values()).index(pred_target_top)]
            pred_proba = self.model.predict_proba([face_encod])[0]
            pred_proba_top = pred_proba[pred_target_top][0]
            #df_name_predict = pd.DataFrame()
            #pred_proba = self.model.pred_proba([face_encod])[0][pred_target][0]

            # print("predict: %d" % pred_target)
            # print("predict name: %s" % pred_name)
            # print(pred_proba)

            return pred_name_top, pred_proba_top

    def __load_data(self) -> tuple[np.array, dict]:
        """ Загрузка данных для обучения """

        try:
            path_model = os.path.join(self.path_load_model, 'model_LR.pkl')
            with open(path_model, 'rb') as file:
                load_model = pickle.load(file)

            path_act = os.path.join(self.path_load_model, 'name_targets.json')
            with open(path_act, 'r') as file:
                load_name_targets = json.load(file)
        except Exception as ex:
            print(f'Error: {ex}')
        else:
            return load_model, load_name_targets

    def __load_image(self) -> np.array:
        """ Загрузка изображения, с изменением размера """

        # path_test_image = os.path.join(self.path_load, 'user_photo.jpg')
        # изменение формата тестового изображения
        with Image.open(self.path_load_image) as photo:
            photo_resized = preprocessing.resize_photo(photo, self.size_new)
            photo_resized_conv = np.array(photo_resized.convert('RGB'))

        return photo_resized_conv


# что за хрень
def __test_predict_model_img_lr():
    path_dir = os.path.abspath(os.path.join('..', *config['predict']['path']))
    user_name = 'dovolno_slov'
    user_photo_name = f'{user_name}_photo.jpg'
    path_load = os.path.join(path_dir, user_name, user_photo_name)

    predict_model_img_lr = PredictModelImgLR(path_load, SIZE_USERS_PHOTO_NEW)
    pred_name, pred_proba_top = predict_model_img_lr.predict_model()

    print("predict name: %s" % pred_name)
    print(pred_proba_top)


if __name__ == "__main__":
    __test_predict_model_img_lr()
