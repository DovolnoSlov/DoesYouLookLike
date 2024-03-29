import numpy as np
import os
import face_recognition
import pickle
import json


class GetEmbedding:
    """
    Поиск лиц на фотографиях, и сохранение полученных эмбедингов в pickle

    Args:
        path_load (str): путь до каталогов с изображениями
        actors (list): список актёров/актрис
        path_save (str): путь сохранения эмбеддингов и таргетов
    """

    def __init__(self, path_load: str, actors: list, path_save: str):
        self.path_load = path_load
        self.actors = actors
        self.path_save = path_save

    def get_save_embedding(self) -> None:
        """ Получение эмбеддингов, таргетов, имён с индексами, и сохранение в файлы """

        embeddings, targets, name_targets = self.__create_embedding()

        path_emb = os.path.join(self.path_save, 'embeddings.pkl')
        with open(path_emb, 'wb') as f:
            pickle.dump(embeddings, f)

        path_tar = os.path.join(self.path_save, 'targets.pkl')
        with open(path_tar, 'wb') as f:
            pickle.dump(targets, f)

        path_act = os.path.join(self.path_save, 'name_targets.json')
        json_act = json.dumps(name_targets, indent=4)
        with open(path_act, 'w') as f:
            f.write(json_act)

# -> tuple[np.array, list, dict]
    def __create_embedding(self):
        """
        Поиск лиц,
        и формирование эмбеддингов, таргетов и словаря имена:таргеты

        :return: эмбеддинги, таргеты, словарь имена:таргеты
        :rtype: tuple[np.array, list, dict]
        """

        embeddings = np.empty(128)
        targets = []
        name_targets = self.__create_targets()
        for name in self.actors:
            path_to_images = os.path.join(self.path_load, name)
            images_for_name = os.listdir(path_to_images)
            for img in images_for_name:
                try:
                    face = self.__load_image(path_to_images, img)
                    if self.__count_face_locations(face):
                        continue

                    face_encod = face_recognition.face_encodings(face)[0]
                    embeddings = np.vstack((embeddings, face_encod))
                    # добавление таргета по имени
                    targets.append(name_targets[name])

                except Exception as ex:
                    print(f'Error: {ex}')

        return embeddings[1:], targets, name_targets

    def __create_targets(self) -> dict:
        """ Создание словаря имена:таргеты """

        name_targets = dict()
        for target, name in enumerate(self.actors):
            name_targets[name] = target
        return name_targets

    def __load_image(self, path_to_images: str, img: str) -> np.array:
        """ Загрузка изображения """

        path_image = os.path.join(path_to_images, img)
        load_face = face_recognition.load_image_file(path_image)
        return load_face

    def __count_face_locations(self, face: np.array) -> bool:
        """ Поиск и проверка количества лиц на изображении """

        face_boxes = face_recognition.face_locations(face)
        # если найдено больше 1 лица на изображении - оно исключается
        if len(face_boxes) != 1:
            return True
        return False
