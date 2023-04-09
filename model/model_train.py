import numpy as np
import pickle
import yaml
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import preprocessing

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

PATH_IMAGES = os.path.abspath(os.path.join('..', *config['load_images']['path']))
TARGET_ACTORS = config['load_images']['images']['target_actors']
NEW_SIZE_OF_IMAGE = config['load_images']['images']['size_new']
LIMIT_LOAD_IMAGES = config['load_images']['images']['limit']

PATH_MODEL = os.path.abspath(os.path.join('..', *config['model']['path']))
RANDOM_STATE = config['model']['random_state']
TEST_SIZE = config['model']['test_size']
COEF_C = config['model']['coef_C']
KEY_LOAD_IMAGES = config['model']['key_load_images']


class NoLoad(Exception):
    pass


class ModelImgLR:
    """
    Модель логистической регрессии

    Args:
        path_load (str): путь до каталога с эмбеддингами и таргетами
        random_state (int): параметр random_state
        test_size (float): параметр test_size для train_test_split
        coef_C (float): параметр C для логистической регрессии
    """

    def __init__(self, path_load: str, random_state: int = 0, test_size: float = 0.3, coef_c: float = 1.0):
        self.path_load = path_load
        self.test_size = test_size
        self.random_state = random_state
        self.coef_c = coef_c

        self.embeddings = None
        self.targets = None
        self.__f1_model_score = None

        # флаг проведения обучения модели
        self.__fit_flag = False

    def fit_model(self) -> None:
        """
        Обучение модели,
        сохранение в pickle формате, в каталог с эмбеддингами и таргетами
        """
        try:
            self.embeddings, self.targets = self.__load_data()
            if not self.embeddings:
                raise NoLoad("Проблема с загрузкой эмбеддингов и таргетов")
        except NoLoad as ex:
            logging.info(f'Произошла ошибка: {ex}')

        min_num_target, name_min_target = self.__check_min_target()
        if min_num_target > 1:
            X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.targets,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                stratify=self.targets)
            model_LR = LogisticRegression(random_state=self.random_state, C=self.coef_c)
            model_LR.fit(X_train, y_train)

            self.__save_model(model_LR)

            self.__f1_model_score = f1_score(y_test, model_LR.predict(X_test), average='micro')
            self.__fit_flag = True

        else:
            logging.info(f'Проблемы с данными. Таргет: {name_min_target}')

    def __load_data(self) -> tuple[np.array, list]:
        """ Загрузка данных для обучения """

        try:
            path_embeddings = os.path.join(self.path_load, 'embeddings.pkl')
            with open(path_embeddings, 'rb') as file:
                load_embeddings = pickle.load(file)

            path_targets = os.path.join(self.path_load, 'targets.pkl')
            with open(path_targets, 'rb') as file:
                load_targets = pickle.load(file)
        except Exception as ex:
            logging.info(f'Произошла ошибка: {ex}')
        else:
            return load_embeddings, load_targets

    def __check_min_target(self) -> tuple[int, str]:
        """ Подсчёт количества каждой из меток в списке, с нахождением минимального """

        targets_counter = Counter(self.targets)
        min_num_target = np.inf
        name_min_target = ''
        for target in targets_counter.keys():
            if targets_counter[target] < min_num_target:
                min_num_target = targets_counter[target]
                name_min_target = target

        return min_num_target, str(name_min_target)

    def __save_model(self, model_LR: LogisticRegression) -> None:
        """ Сохранение модели логистической регрессии в файл """

        path_save = os.path.join(self.path_load, 'model_LR.pkl')
        with open(path_save, 'wb') as f:
            pickle.dump(model_LR, f)

    def get_score(self):
        """ Получение значения по метрике f1_score """

        if not self.__fit_flag:
            logging.info(f'Модель не обучена!')
        return self.__f1_model_score


def __main():
    if KEY_LOAD_IMAGES:
        # загрузка изображений указанных актёров/актрис
        preprocessing.download_images(PATH_IMAGES, TARGET_ACTORS, LIMIT_LOAD_IMAGES)

        # изменение размера всех изображений
        preprocessing.reformat_image(PATH_IMAGES, TARGET_ACTORS, NEW_SIZE_OF_IMAGE)

        # моздание объекта класса, для поиска лиц на фотографиях
        actors_embedding = preprocessing.GetEmbedding(PATH_IMAGES, TARGET_ACTORS, PATH_MODEL)

        # получение эмбеддингов, таргетов, имён с индексами, и сохранение в файлы
        actors_embedding.get_save_embedding()

    my_model = ModelImgLR(PATH_MODEL, RANDOM_STATE, TEST_SIZE, COEF_C)
    my_model.fit_model()
    f1_model_score = my_model.get_score()
    print(f'Метрика модели: F1 score: {f1_model_score}')


if __name__ == "__main__":
    __main()

