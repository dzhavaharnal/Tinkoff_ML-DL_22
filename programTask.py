import numpy as np 
import argparse
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import load_model



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir') #путь к директории, в которой лежит коллекция документов
parser.add_argument('--model', required = True) #путь к файлу, в который сохраняется модель
parser.add_argument('--prefix') #начало предложения (одно или несколько слов)
parser.add_argument('--length', required = True, type = int) #длина генерируемой последовательности
args = parser.parse_args()

########## Обучение ##########

#Считываем входные данные из файла либо с потока ввода
if args.input_dir: #Если  аргумент не задан, считаем, что тексты вводятся из stdin
    with open(args.input_dir) as f:
        data = f.read()
else:
    data = sys.stdin.read()


corpus = data.lower().split("\n")

#Делаем токенизацию
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


#Создаем n-граммы
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])

#делаем padding, выранивая длину
padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

#получаем обучающую выборку, делаем категоризацию
features = input_sequences[:,:-1]
labels = input_sequences[:,-1]
one_hot_labels = to_categorical(labels, num_classes=total_words)



#Закончим обучение, если точность выше 0.95
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.95:
                self.model.stop_training = True
callbacks = myCallback()
#строим модель двунаправленную LSTM
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#обучаем модель
history = model.fit(features, one_hot_labels, epochs=500, callbacks=[callbacks])
#сохраняем модель
model.save(args.model)  


########## Генерация ##########

model = load_model(args.model)

#Начало предложения (одно или несколько слов). Если не указано, выбираем начальное слово случайно из всех слов
if args.prefix: 
   seed_text = args.prefix
else:
    seed_text = list(tokenizer.word_index.keys())[np.random.randint(0,len(tokenizer.word_index))] 

next_words = args.length

#Генерируем последовательность заданной длины и выводим её на экран
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted = np.argmax(predicted, axis=-1).item()
    output_word = tokenizer.index_word[predicted]
    seed_text += " " + output_word

print(seed_text)