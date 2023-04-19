# DoesYouLookLike
Who do you look like from celebrities


Чат-бот для телеграм, который по фотографии, с применением логистической регрессии, делает предсказание схожести со звездой голливуда!

В проект входит:
- загрузка изображений знаменитостей из поисковика Bing
- обработка изображений, создание данных на их основе
- обучение модели логистической регресси (запускается вручную)
- сохранение модели после обучения в файл, для многократного использования 
- реализация чат-бота, который использует полученную модель для предсказаний на основе изображения пользователя
- конфигурационный файл, который позволяет варьировать некоторые параметры

Описание файлов проекта:

	notebook\DoesYouLookLike_JN.ipynb - файл с первоначальным кодом на получение и обработку изображений, данных и самой модели, с последующим предсказанием на тестовом изображении.

preprocessing:
	load_images.py:
		download_images - загрузка изображений из Bing по запросу, сохранение в каталоге data\train_images
				rename_dir - изменение имени каталогов с изображениями, сформированного по тексту запроса
				count_files_in_dir - подсчёт количества файлов в каталогах с условием <2 -> True 
				reformat_image - изменение размерности изображений, по списку имён, с помощью resize_image
				resize_image - изменение размерности изображения (х1), поступающего на вход
		create_data.py:
				class GetEmbedding - класс для получения эмбеддингов и таргетов после обработки изображений
				
				get_save_embedding - главный метод класса, который получает данные из скрытых методов класса, и сохраняет результаты в каталоге data\model_data


	model:
		model_train.py:
			class ModelImgLR: - класс для создания и обучения модели, с сохранением модели в каталоге data\model_data
				+проверка ключа KEY_LOAD_TRAIN_IMAGES из config_model.yaml для оценки необходимости загрузки тренировочных изображений
		model_predict.py:
		 	class PredictModelImgLR - класс для предсказания по модели на основе изображения пользователя. 
		 		predict_model - главный метод класса, который получает данные из скрытых методов класса, и выдаёт результат предсказания

	trial_iteration\trial_iteration.py - пробная итерация предсказания по готовой обученной модели. Тест по своей сути

Дополнительная информация по последним обновлениям:
- в данный момент модель обучена на лицах только женского пола
- в код включено логгирование, которое позволяет отслеживать работу чат-бота в онлайн-режиме
