import yaml
import os
from model_class import ModelImgLR
import preprocessing

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

pathLoad = os.path.abspath(os.path.join('.', *config['load_data']['path']))
targetActors = config['load_data']['images']['target_actors']
sizeImageNew = config['load_data']['images']['size_new']
limitLoadImage = config['load_data']['images']['limit']

pathModel = os.path.abspath(os.path.join('.', *config['model']['path_data']))
randomState = config['model']['random_state']
testSize = config['model']['test_size']
coefC = config['model']['coef_C']
keyLoadImages = config['model']['key_load_images']


MyModel = ModelImgLR(path_model, random_state, test_size, coef_C)

model_LR, X_test, y_test = MyModel.fit_model()
f1_model_score = f1_score(y_test, model_LR.predict(X_test), average='micro')


print(f'F1 score: {f1_model_score}')

if __name__ == "__main__":
    pass

