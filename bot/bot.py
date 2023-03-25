import config
import logging

from aiogram import Bot, Dispatcher, executor, types

# log level
logging.basicConfig(level=logging.INFO)

# bot init
bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


# Hi, name
@dp.message_handler(regexp='(^[Hh]i|^[Пп]ривет)')
async def echo(message: types.Message):
    user_full_name = message.from_user.full_name
    answer_text = 'Привет, {name}!' \
                  '\nПрекрасно выглядишь!'.format(name=user_full_name)
    await message.answer(answer_text)


@dp.message_handler(content_types=['document', 'photo'])
async def download_photo(message: types.Message):
    if message.content_type == 'document':
        inp_photo = message.document
    if message.content_type == 'photo':
        inp_photo = message.photo[-1]
    user = message.from_user.full_name
    await inp_photo.download(destination_file=f"../user_photo/{user}/{inp_photo.file_id}.jpg")


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
