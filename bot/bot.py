import config_bot
import logging
import os
import yaml
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext

# log level
logging.basicConfig(level=logging.INFO)

__config_path = os.path.abspath(os.path.join('..', 'config', 'config_model.yaml'))
with open(os.path.join(__config_path)) as f:
    config = yaml.safe_load(f)

LOAD_USER_PHOTO_PATH = os.path.abspath(os.path.join('..', *config['predict']['path']))

# bot init
bot = Bot(token=config_bot.TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


class UserState(StatesGroup):
    gender = State()    # на будущее, для датасетов разных полов
    photo = State()


@dp.message_handler(commands=['start'])
async def user_register(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['М', 'Ж']
    keyboard.add(*buttons)

    user_full_name = message.from_user.full_name
    answer_text = "Привет, {name}!" \
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
        inp_photo, load_path = await __doc_type_path(message)
        await inp_photo.download(destination_file=load_path)
    else:
        answer = "К сожалению, это не фотография. Попробуйте сновая, начиная с команды /start"
        await bot.send_message(message.from_user.id, answer)
    logging.info(f"Закачка фото, финал")
    await state.finish()


async def __doc_type_path(message):
    if message.content_type == 'document':
        inp_photo = message.document
    elif message.content_type == 'photo':
        inp_photo = message.photo[-1]
    user_name = message.from_user.username
    user_photo_name = f'{user_name}_photo.jpg'
    load_path = os.path.join(LOAD_USER_PHOTO_PATH, user_name, user_photo_name)
    logging.info(f"Закачка фото, этап 2")
    return inp_photo, load_path


# Hi, name
@dp.message_handler(regexp='(^[Hh]i|^[Пп]ривет)')
async def echo(message: types.Message):
    user_full_name = message.from_user.full_name
    answer_text = "Привет, {name}!" \
                  "\nПрекрасно выглядишь!".format(name=user_full_name)
    await message.answer(answer_text)


@dp.message_handler(content_types=['document', 'photo'])
async def download_photo(message: types.Message):
    if message.content_type == 'document':
        inp_photo = message.document
    if message.content_type == 'photo':
        inp_photo = message.photo[-1]
    user_name = message.from_user.username
    user_photo_name = f'{user_name}_photo.jpg'
    load_path = os.path.join(LOAD_USER_PHOTO_PATH, user_name, user_photo_name)
    await inp_photo.download(destination_file=load_path)


    # if message.content_type == 'document':
    #     file_id = message.document.file_id
    # if message.content_type == 'photo':
    #     file_id = message.photo[-1].file_id
    #
    # file = await bot.get_file(file_id)
    # file_path = file.file_path
    # await bot.download_file(file_path, destination=f"../user_photo/123/123.jpg")


# cat's foto
@dp.message_handler(regexp='(^cat[s]?$|puss|seba|Seba)')
async def cats(message: types.Message):
    with open('seba/seba_001.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='Cats are here 😺')

# polina's foto
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
