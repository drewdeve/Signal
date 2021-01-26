import numpy as np
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
import sqlite3
import sys
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import random
import pickle
import os
import threading
from  optparse import OptionParser
import msvcrt

# настройка tensorflow
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
#config = tf.ConfigProto(
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#    # device_count = {'GPU': 1}
#    )
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#set_session(session)


# следующие переменные настраивают систему
Fc = 100  # имитировать несущую частоту 1 кГц
Fbit = 50  # смоделированный битрейт данных
Fdev = 5  # отклонение частоты, сделать больше, чем битрейт
N = 64  # сколько бит отправить
A = 1  # амплитуда передаваемого сигнала
Fs = 1000  # частота дискретизации для симулятора, должна быть выше, чем в два раза несущая частота
A_n = 0.70  # пиковая амплитуда шума
N_prntbits = 10  # количество бит для печати в графиках
db_path = "train.db" # путь  к базе  образцов сигналов
researh_signal = [1,1,0] # иследуемый сигнал
epoch=100

# создаем массив типов сигналов для обучения
m_signals = [[1,1,1],
             [1,0,1],
             [1,1,0],
             [1,0,0],
             [0,1,0]]


#Генерация сигнала input_ds - массив трех бит [b,b,b], если plot True то вывод при помощи matplotlib.pylab
def generate_signal(input_ds):
    global t
    global m
    global N_prntbits 
    arr = []
    for bit in input_ds:
        arr = np.concatenate([arr, np.full((10,), bit, dtype=int)])
    data_in = arr
    N = N_prntbits = len(data_in)
    """
    VCO
    """  
    t = np.arange(0, float(N) / float(Fbit), 1 / float(Fs),
                  dtype=np.float)
    
    # расширить data_in, чтобы учесть битрейт и преобразовать 0/1 в частоту
    m = np.zeros(0).astype(float)
    for bit in data_in:    
        if bit == 1:
            m = np.hstack((m, np.multiply(np.ones(Fs // Fbit), Fc + Fdev)))
        else:
            m = np.hstack((m, np.multiply(np.ones(Fs // Fbit), Fc - Fdev)))
    # рассчитать выход VCO
    n=A*10**(A_n)
    noise = np.random.uniform(0,n,600)

    y = np.zeros(0)  
    y = A * np.cos(2 * np.pi * np.multiply(m, t)+noise)
    return y

def plot_signal(y):
    # смотрим данные во временной и частотной состовляющих
    # рассчитать частотную состовляющую для просмотра
    N_FFT = float(len(y))
    f = np.arange(0, Fs / 2, Fs / N_FFT)  
    y_f =abs( np.fft.fft(y))
    pl.figure(figsize=(17, 7))
    pl.subplot(3, 1, 1)
    pl.plot(t[0: int(Fs * N_prntbits / Fbit)], m[0: (Fs * N_prntbits // Fbit)])
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Original VCO output versus time')
    pl.grid(True)
    pl.subplot(3, 1, 2)
    pl.plot(t[0:int(Fs * N_prntbits / Fbit)], y[0: int(Fs * N_prntbits / Fbit)], linewidth=1)
    pl.xlabel('Time (s)')
    pl.ylabel('Amplitude (V)')
    pl.title('Amplitude of carrier versus time')
    pl.grid(True)
    pl.subplot(3, 1, 3)
    pl.plot(f[0: int(Fc + Fdev * 2 * N_FFT / Fs)], y_f[0:int(Fc + Fdev * 2 * N_FFT / Fs)])
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Amplitude (dB)')
    pl.title('Spectrum')
    pl.grid(True)
    pl.tight_layout()        
    pl.show()      
        
    

# функция для чтения сигналов из базы данных 
def read_train_base():    
    connection=None
    try:
        # соединяемся с БД считываем весь набор данных
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        sql = "SELECT data, type FROM signals"
        cursor.execute(sql)
        dbset = cursor.fetchall()
        # перемешиваем данные
        random.seed(50)
        random.shuffle(dbset)

        # обходим данные и записывем попарно в массивы data (данные сигнала) и labels( его метка)
        data = []
        labels=[]
        for s in dbset:
            data.append(np.frombuffer(s[0]).tolist())
            labels.append(s[1])

        data = np.array(data, dtype="float")
        labels = np.array(labels)
        return (data,labels)

    except :
        print (sys.exc_info()) # Выводим сообщение о ошибке

    if(connection):
        connection.close()


# Обучение модели на основе сигналов из БД 
def train_model():    
    data = []
    labels=[]
    # считываем сигнали и их метки
    (data,labels) = read_train_base()
    # делим набор данных на данные для обучения(99%) и данные для тестирования (1%)
    (train_data, test_data, train_labels, test_labels) = train_test_split(data,labels, test_size=0.01, random_state=50)   
    
    # определение структуры модели
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(600,)))
    model.add(keras.layers.Dense(300, activation="relu"))   
    model.add(keras.layers.Dense(150, activation="relu"))    
    model.add(keras.layers.Dense(len(m_signals), activation="softmax"))
    model.summary()
    print("Обучение сети...")
    
    # компилирование модели
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=epoch)
    print("Успешно")

    # оценка точности на проверочных данных
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Точность на проверочных данных:', test_acc)

    # сохранение модели и массив типов на диск
    print("Сохранение модели и массив типов на диск...")
    model.save("model_ffnn.h5")
    f = open("labels_ffnn.bin", "wb")
    f.write(pickle.dumps(m_signals))
    f.close()
    print("Успешно")
    f = open("train_history_ffnn.bin", "wb")
    f.write(pickle.dumps(history.history['loss']))
    f.close()

def predictions(signal):
    print("Загрузка модели нейронной сети...")
    try:
        model = keras.models.load_model("model_ffnn.h5")
        model.summary()        
        lb = pickle.loads(open("labels_ffnn.bin", "rb").read())
        print("Успешно")
          
        #Определание типа сигнала

        pred = model.predict(signal)
        index_label=int(np.argmax(pred[0]))
        label=lb[index_label]
        print("Нейросеть определила, что тип иследуемого сигнала ", label)

    except :        
        print("Ошибка загрузки модели")

  


def main():
    global epoch
    usage = "Использование: %prog [опции] арг"
    parser = OptionParser(usage)
   
    parser.add_option("-t", "--train",default=False, dest="train_nn",
                      help="Обучение нейронной сети на основе БД сигналов. По умолчанию: False")   
    parser.add_option("-e", "--epoch",default=100, dest="epoh_nn",type='int',
                      help="Количество эпох. По умолчанию:100 ")
    (options, args) = parser.parse_args()
    try:
        epoch = int(options.epoh_nn)        
    except :
        pass
    
    if options.train_nn:
        train_model()
        #msvcrt.getch()
    else:
        array_signal = generate_signal(researh_signal)
        result = (np.expand_dims(array_signal,0))
        print(result)
        predictions(result)
        plot_signal(array_signal)


if __name__ == "__main__":
    main()