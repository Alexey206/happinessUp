pip install pytelegrambotapi
import telebot
import project

happinessBot = telebot.TeleBot('1300453094:AAG9LiX-DQ0d1SKw-Hiys0kajz2XYt5uGg4')

keyboard = telebot.types.ReplyKeyboardMarkup(True)
keyboard.row('ВВП СНГ', 'ВВП Евросоюза', 'ВВП Азии', 'ВВП Америки', 'Счастье СНГ', 'Счастье Евросоюза', 'Счастье Азии', 'Счастье Америки')

@happinessBot.message_handler(commands=['start'])
def start_message(message):
  happinessBot.send_message(message.chat.id, 'Приветствую, я — бот команды HappinessUp. Я умею выдавать ВВП и индекс счастья стран СНГ, Евросоюза, Азии и Америки.', reply_markup=keyboard)

@happinessBot.message_handler(content_types=['text'])
def send_text(message):
  if message.text == 'ВВП СНГ':
    happinessBot.send_message(message.chat.id, 'ВВП СНГ: \n{}'.format(GDP_cis[['Country', 2020]]))
  elif message.text == 'ВВП Евросоюза':
    happinessBot.send_message(message.chat.id, 'ВВП Евросоюза: \n{}'.format(GDP_eu[['Country', 2020]]))
  elif message.text == 'ВВП Азии':
    happinessBot.send_message(message.chat.id, 'ВВП азиатских государств: \n{}'.format(GDP_asia[['Country', 2020]]))
  elif message.text == 'ВВП Америки':
    happinessBot.send_message(message.chat.id, 'ВВП стран американского континента: \n{}'.format(GDP_america[['Country', 2020]]))
  elif message.text == 'Счастье СНГ':
    happinessBot.send_message(message.chat.id, 'Индекс счастья СНГ: \n{}'.format(cis_analytics20))
  elif message.text == 'Счастье Евросоюза':
    happinessBot.send_message(message.chat.id, 'Индекс счастья Евросоюза: \n{}'.format(eu_analytics20))
  elif message.text == 'Счастье Азии':
    happinessBot.send_message(message.chat.id, 'Индекс счастья азиатских государств: \n{}'.format(asia_analytics20))
  elif message.text == 'Счастье Америки':
    happinessBot.send_message(message.chat.id, 'Индекс счастья стран американского континента: \n{}'.format(america_analytics20))

happinessBot.polling()
