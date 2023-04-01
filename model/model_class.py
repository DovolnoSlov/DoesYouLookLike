import numpy as np
import os
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class ModelImgLR:
    """
    Модель логистической регрессии


    Args:
        path_load (str): путь до каталога с эмбеддингами и таргетами
        random_state (int): параметр random_state
        test_size (float): параметр test_size для train_test_split
        coef_C (float): параметр C для логистической регрессии
    """

    def __init__(self, path_load: str, random_state: int = 0, test_size: float = 0.3, coef_C: float = 1.0):
        self.path_load = path_load
        self.test_size = test_size
        self.random_state = random_state
        self.coef_C = coef_C

        # загрузка эмбеддингов и таргетов
        self.embeddings, self.targets = self.__load_data()

    def fit_model(self) -> LogisticRegression:
        """
        Обучение модели,
        сохранение в pickle, в каталог с эмбеддингами и таргетами

        :return: модель машинного обучения, тестовые данные X_test и y_test
        :rtype: LogisticRegression, np.array, np.array

        """

        min_num_target, name_min_target = self.__check_min_target()
        if min_num_target > 1:
            X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.targets,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                stratify=self.targets)
            model_LR = LogisticRegression(random_state=self.random_state, C=self.coef_C)
            model_LR.fit(X_train, y_train)

            self.__save_model(model_LR)

            return model_LR, X_test, y_test
        else:
            # logging.info(f'Problem with data. Target: {name_min_target}')
            print(f'Problem with data. Target: {name_min_target}')

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
            print(f'Error: {ex}')
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