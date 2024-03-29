from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext
import os
import yaml
from model import model_predict
from dotenv import load_dotenv

load_dotenv()

# log level
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

__config_path = os.path.abspath(os.path.join('config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

PATH_SAVE_USER_IMAGE = os.path.abspath(os.path.join(*config['predict']['path']))

# bot init
TOKEN = os.getenv("TOKEN")
bot = Bot(token=TOKEN)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


class UserState(StatesGroup):
    gender = State()    # на будущее, для доп. функционала
    photo = State()


@dp.message_handler(commands=['start', 'help'])
async def send_help(message: types.Message):
    bot_name = await bot.get_me()
    await message.reply(f"Меня зовут {bot_name['username']}, "
                        f"приятно познакомиться!\n"
                        f"/like - по данной команде Вы запустите работу, "
                        f"в рамках которой нужно будет отправить фотографию, "
                        f"и Вы получите результат схожести с кем-то из знаменитых людей!"
                        f"/mood - по данной команде происходит оперативное поднятие настроения!")


@dp.message_handler(commands=['like'])
async def like_get_gender(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['М', 'Ж']
    keyboard.add(*buttons)

    user_full_name = message.from_user.full_name
    answer_text = "Начнём, {name}!" \
                  "\nукажите свой пол".format(name=user_full_name)

    logging.info(f'Набор команды like, пользователь: {message.from_user.username}')
    await message.answer(answer_text, reply_markup=keyboard)
    await UserState.gender.set()


@dp.message_handler(state=UserState.gender)
async def like_get_photo(message: types.Message, state: FSMContext):
    await state.update_data(user_gender=message.text)
    await message.answer("Отлично! Теперь отправьте фотографию", reply_markup=types.ReplyKeyboardRemove())
    logging.info(f'Пол выбран')
    await UserState.photo.set()


@dp.message_handler(state=UserState.photo, content_types=['document', 'photo', 'text'])
async def like_result(message: types.Message, state: FSMContext):
    if message.content_type in ['document', 'photo']:
        inp_photo, path_save = await __doc_type_path(message)
        await inp_photo.download(destination_file=path_save)
        await message.reply('Фото получено. Пожалуйста, ожидайте результат.')
        logging.info("Загрузка изображения пользователя, завершено")

        botmodel = model_predict.PredictModelImgLR(path_save)
        answer_pred = botmodel.predict_model()
        await message.answer(answer_pred)
        logging.info(f"Изображение пользователя {message.from_user.username} обработано, ответ направлен")
    else:
        answer = "К сожалению, это не фотография. Попробуйте снова, начиная с команды /like"
        await message.reply(answer)
        logging.info(f"Получено не изображение. Работа с пользователем {message.from_user.username} завершена")

    await state.finish()


async def __doc_type_path(message):
    if message.content_type == 'document':
        # if message.document.filename .jpg, .png, ....: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        inp_photo = message.document
    elif message.content_type == 'photo':
        inp_photo = message.photo[-1]
    user_name = message.from_user.username
    user_photo_name = f'{user_name}_photo.jpg'
    path_save = os.path.join(PATH_SAVE_USER_IMAGE, user_name, user_photo_name)
    logging.info(f"Загрузка изображения пользователя, оценка типа и создание пути")
    return inp_photo, path_save


# Hi, name
@dp.message_handler(regexp='(^[Hh]i|^[Пп]ривет)')
async def hello_answer(message: types.Message):
    user_full_name = message.from_user.full_name
    answer_text = "Привет, {name}!" \
                  "\nПрекрасно выглядишь!".format(name=user_full_name)
    await message.answer(answer_text)


# creating a mood
@dp.message_handler(commands=['mood'])
@dp.message_handler(regexp='(^[Pp]olina|[Pp]oly|[Пп]олина|[Пп]оля)')
async def create_mood(message: types.Message):
    with open('bot/raznoe/beautiful_in_the_world.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='😉')


# secret cat's foto
@dp.message_handler(regexp='(^[Cc]at[s]?$|^[Pp]uss|^[Ss]eba|^[Сс]еба)')
async def secret_cat(message: types.Message):
    with open('bot/seba/seba_001.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='Cats are here 😺')


# echo
@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


# run long-poling
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
