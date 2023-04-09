__all__ = ['download_images', 'rename_dir', 'count_files_in_dir',
           'reformat_image', 'resize_image']

import os
import shutil
from bing_image_downloader.downloader import download
from PIL import Image


def download_images(path_download: str, actors: list, limit_load: int = 15) -> None:
    """
    Загрузка изображений указанных актёров/актрис

    :param path_download: путь для сохранения изображений на сервере
    :param actors: список актёров/актрис
    :param limit_load: количество изображений для загрузки

    :return: None
    """

    # проверка наличия каталогов с изображениями
    # удаление в случае нахождения
    try:
        shutil.rmtree(path_download)
        print('Дерево каталогов удалено')
    except OSError as e:
        print('Error: %s : %s' % (path_download, e.strerror))

    # загрузка изображений из Bing
    for name_actor in actors:
        find_string = f'face {name_actor}'
        download(find_string, limit=limit_load, output_dir=path_download,
                 adult_filter_off=True, force_replace=False, timeout=60, verbose=False)

        # переименование каталогов с загруженными изображениями
        rename_dir(path_download, find_string, name_actor)

        # подсчёт количества загруженных изображений по каждому запросу
        # с удалением при количестве < 2
        if count_files_in_dir(path_download, name_actor):
            actors.remove(name_actor)


def rename_dir(path_dir: str, name_old: str, name_new: str) -> None:
    """
    Переименование каталога

    :param path_dir: путь хранения каталогов
    :param name_old: старое имя каталога
    :param name_new: новое имя каталога

    :return: None
    """

    path_old = os.path.join(path_dir, name_old)
    path_new = os.path.join(path_dir, name_new)
    os.rename(path_old, path_new)


def count_files_in_dir(path_dir: str, name_dir: str) -> bool:
    """
    Оценка количества файлов в каталогах

    :param path_dir: путь хранения каталогов
    :param name_dir: наименование проверяемого каталога

    :return: True (if < 2) / False
    :rtype: bool
    """

    path_listdir = os.path.join(path_dir, name_dir)
    files = len(os.listdir(path_listdir))
    if files < 2:
        return True
    return False


def reformat_image(path_load: str, actors: list, size_new: int) -> None:
    """
    Изменение размера всех изображений

    :param path_load: путь до каталогов с изображениями
    :param actors: список актёров/актрис
    :param size_new: необходимый размер изображения по одной из сторон

    :return: None
    """

    for name in actors:
        path_to_images = os.path.join(path_load, name)
        images = os.listdir(path_to_images)
        for img in images:
            path_image = os.path.join(path_to_images, img)
            # изменение формата изображения и сохранения под тем же именем
            with Image.open(path_image) as image:
                image_resized = resize_image(image, size_new)
                image_resized_conv = image_resized.convert('RGB')
                image_resized_conv.save(path_image)


def resize_image(image: Image, size_new: int) -> Image:
    """
    Изменение размера изображения

    :param image: исходное изображение
    :param size_new: необходимый размер изображения по одной из сторон

    :return: финальное изображение
    :rtype: Image
    """

    # получение размера исходного изображения
    size = image.size

    # рассчёт коэффициента по одной из сторон
    coef = size_new / size[0]
    first_side = int(size[0] * coef)
    second_side = int(size[1] * coef)

    # изменение размера изображения
    resized_image = image.resize((first_side, second_side))
    resized_image = resized_image.convert('RGB')
    return resized_image
