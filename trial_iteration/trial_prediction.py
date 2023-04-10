import os
import yaml
from model import model_predict

__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

PATH_LOAD_MODEL = os.path.abspath(os.path.join('..', *config['model']['path']))


def __trial_predict_model_img_lr():
    """ Пробная итерация предсказания модели на тестовом изображении """

    path_load_image = os.path.join('.', 'trial_image_JR.jpg')
    predict_model_img_lr = model_predict.PredictModelImgLR(path_load_image=path_load_image,
                                                           size_new=512,
                                                           path_load_model=PATH_LOAD_MODEL)
    answer_pred = predict_model_img_lr.predict_model()

    if 'Julia Roberts' in answer_pred:
        test = True
    else:
        test = False
    print(f'Тест: {test}\n')
    print(answer_pred)


if __name__ == "__main__":
    __trial_predict_model_img_lr()