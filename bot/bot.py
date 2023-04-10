from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext
import os
import yaml
from model import model_predict

import config_bot   # содержит токен чат-бота

# log level
import logging
logging.basicConfig(level=logging.INFO)

__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

PATH_SAVE_USER_IMAGE = os.path.abspath(os.path.join('..', *config['predict']['path']))

# bot init
bot = Bot(token=config_bot.TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


class UserState(StatesGroup):
    gender = State()    # на будущее, для датасетов разных полов
    photo = State()


@dp.message_handler(commands=['start', 'help'])
async def send_help(message: types.Message):
    bot_name = await bot.get_me()
    await message.reply(f"Меня зовут {bot_name['username']}, "
                        f"приятно познакомиться!\n"
                        f"/like - по данной команде Вы запустите работу, "
                        f"в рамках которой нужно будет отправить фотографию, "
                        f"и Вы получите результат схожести с кем-то из знаменитых людей!")


@dp.message_handler(commands=['like'])
async def user_register(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['М', 'Ж']
    keyboard.add(*buttons)

    user_full_name = message.from_user.full_name
    answer_text = "Начнём, {name}!" \
                  "\nукажите свой пол".format(name=user_full_name)

    logging.info(f'Набор команды start')
    await message.answer(answer_text, reply_markup=keyboard)
    await UserState.gender.set()


@dp.message_handler(state=UserState.gender)
async def get_username(message: types.Message, state: FSMContext):
    await state.update_data(user_gender=message.text)
    await message.answer("Отлично! Теперь отправьте фотографию", reply_markup=types.ReplyKeyboardRemove())
    logging.info(f'Пол выбран')
    await UserState.photo.set()


@dp.message_handler(state=UserState.photo, content_types=['document', 'photo', 'text'])
async def get_address(message: types.Message, state: FSMContext):
    if message.content_type in ['document', 'photo']:
        inp_photo, path_save = await __doc_type_path(message)
        await inp_photo.download(destination_file=path_save)
        await message.reply('Фото получено. Пожалуйста, ожидайте результат.')
        logging.info("Закачка фото, финал")

        model = model_predict.PredictModelImgLR(path_save)
        answer_pred = await model.predict_model()
        await message.answer(answer_pred)
        logging.info("Обработано, ответ направлен")
    else:
        answer = "К сожалению, это не фотография. Попробуйте снова, начиная с команды /start"
        await message.reply(answer)
        logging.info("Получено не изображение")

    await state.finish()


async def __doc_type_path(message):
    if message.content_type == 'document':
        inp_photo = message.document
    elif message.content_type == 'photo':
        inp_photo = message.photo[-1]
    user_name = message.from_user.username
    user_photo_name = f'{user_name}_photo.jpg'
    path_save = os.path.join(PATH_SAVE_USER_IMAGE, user_name, user_photo_name)
    logging.info(f"Закачка фото, этап 2")
    return inp_photo, path_save


# Hi, name
@dp.message_handler(regexp='(^[Hh]i|^[Пп]ривет)')
async def echo(message: types.Message):
    user_full_name = message.from_user.full_name
    answer_text = "Привет, {name}!" \
                  "\nПрекрасно выглядишь!".format(name=user_full_name)
    await message.answer(answer_text)


# cat's foto
@dp.message_handler(regexp='(^cat[s]?$|puss|seba|Seba)')
async def cats(message: types.Message):
    with open('seba/seba_001.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='Cats are here 😺')


# Polina's foto
@dp.message_handler(regexp='(^polina|Полина|Поля)')
async def cats(message: types.Message):
    with open('raznoe/beautiful_in_the_world.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='😉')


# echo
@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


# run long-poling
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
