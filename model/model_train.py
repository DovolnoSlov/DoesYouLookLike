import yaml
import os
from model_class import ModelImgLR
import preprocessing

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

size
# os.path.abspath(os.path.join('..', 'data', 'train_images'))
# os.path.abspath(os.path.join('.', *path))


if __name__ == "__main__":
    pass


''' запуск всего скрипта на обучение, файлы до этого не запускаются сами '''
''' С проверкой на необходимость закачки и предобработки данных! '''