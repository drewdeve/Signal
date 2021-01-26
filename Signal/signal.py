from  optparse import OptionParser
import msvcrt
import pickle
import os
import sqlite3
import sys
import numpy as np
import pylab as pl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib . pyplot import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

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
db=7  #    воздействие шума
type_nn = "Рекурентная"
n_epoh=10 # Количество эпох

# создаем массив типов сигналов для обучения
m_signals = [[1,1,1],
             [1,0,1],
             [1,1,0],
             [1,0,0],
             [0,1,0]]

class Mwindow:

    def __init__(self, master):
        self.master = master   
        self.frame = tk.Frame(master)
       

        self.button = tk.Button (master, text="Отобразить", command=self.show)
        self.button.grid(row=0, column=2)

        self.sel_frame = tk.LabelFrame(master,text="Выбор нейронной сети")
        select_list=["Рекурентная","Сверточная","Прямого распостранения"]
        self.combobox_select_type_nn = ttk.Combobox(self.sel_frame,values = select_list)
        self.combobox_select_type_nn.set(str(type_nn))
        self.combobox_select_type_nn.pack()
        self.sel_frame.grid(row=1, column=2,sticky="nwe")

        self.sel_bit_frame = tk.LabelFrame(master,text="Ввод битовой комбинации")
        self.n_bit_frame=tk.Frame(self.sel_bit_frame)
        self.bit1 = tk.StringVar()
        self.bit1.set(str(researh_signal[0]))
        self.b1 = tk.Entry(self.n_bit_frame,textvariable=self.bit1,width=1)
        self.b1.pack(side=tk.LEFT)
        self.bit2 = tk.StringVar()
        self.bit2.set(str(researh_signal[1]))
        self.b2 = tk.Entry(self.n_bit_frame,textvariable=self.bit2,width=1)
        self.b2.pack(side=tk.LEFT)
        self.bit3 = tk.StringVar()
        self.bit3.set(str(researh_signal[2]))
        self.b3 = tk.Entry(self.n_bit_frame,textvariable=self.bit3,width=1)
        self.b3.pack(side=tk.LEFT)
        self.n_bit_frame.pack()
        self.sel_bit_frame.grid(row=2, column=2,sticky="nwe")

        self.sel_noise_db_frame = tk.LabelFrame(master,text="Ввод шума в дц")  
        self.noise_db = tk.StringVar()
        self.noise_db.set(str(db))
        self.noise_db_ent = tk.Entry(self.sel_noise_db_frame,textvariable=self.noise_db,width=2)
        self.noise_db_ent.pack()
        self.sel_noise_db_frame.grid(row=3, column=2,sticky="nwe")

        self.sel_epoh_frame = tk.LabelFrame(master,text="Ввод количества эпох")  
        self.epoh = tk.StringVar()
        self.epoh.set(str(n_epoh))
        self.epoh_ent = tk.Entry(self.sel_epoh_frame,textvariable=self.epoh,width=4)
        self.epoh_ent.pack()
        self.sel_epoh_frame.grid(row=4, column=2,sticky="nwe")
        self.button_train = tk.Button (master, text="Обучить", command=self.train)
        self.button_train.grid(row=5, column=2)
        self.button_graphic = tk.Button (master, text="Показать график", command=self.graphic)
        self.button_graphic.grid(row=7, column=2)

        self.frame.grid() 

    # def graphic():
    #     # Создадим новую фигуру
    #     figure = plt.figure()

    #     # Создадим поле для рисования и получим его оси
    #     axes = figure.add_subplot (1, 1, 1)

    #     # Добавление графика
    #     plt.plot (xlist, ylist)

    #     plt.show()


    #     # Будем рисовать график этой функции
    #     def func (x):
    #         """
    #         sinc (x)
    #         """
    #         if x == 0:
    #             return 1.0
    #         return np.abs (np.sin (x) / x) * np.exp (-np.abs (x / 10))

    #     # Интервал изменения переменной по оси X
    #     xmin = -50.0
    #     xmax = 50.0

    #     # Шаг между точками
    #     dx = 0.1

    #     # Создадим список координат по оси X на отрезке [xmin; xmax)
    #     xlist = np.arange (xmin, xmax, dx)

    #     # Вычислим значение функции в заданных точках
    #     ylist = [func (x) for x in xlist]



        def train (self):        
            global db
            global type_nn
            global n_epoh
        try:
            type_nn=self.combobox_select_type_nn.get()
            print("Выбранный тип сети: ", type_nn)
        except :
            print('Ошибка при выборе типа сети')
        try:
            b1=int(self.bit1.get())
            b2=int(self.bit2.get())
            b3=int(self.bit3.get())
            if((b1==0 or b1==1) and
               (b2==0 or b2==1) and
               (b3==0 or b3==1)):
                researh_signal[0]=b1
                researh_signal[1]=b2
                researh_signal[2]=b3
                print('Битовая комбинация сигнала: ', researh_signal)
            else:
                print('Используется стандартная битовая комбинация сигнала: ', researh_signal)
        except :
            print('Ошибка при вводе битовой комбинации')
            print('Используется стандартная битовая комбинация сигнала: ', researh_signal)

        try:
           db=int(self.noise_db.get())
           print('Установленный шума: ',db)
        except :
            print('Ошибка при вводе шума')

        try:
           n_epoh=int(self.epoh.get())
           print('Установленное количества эпох: ',n_epoh)
        except :
            print('Ошибка при вводе количества эпох')
        create_train_base()
        if type_nn.strip() =="Рекурентная":
            os.system("signal_rnn.py -t=True -e %i"%n_epoh)
            try:
                loss = pickle.loads(open("train_history_rnn.bin", "rb").read())
                ep = range(1, len(loss)+1)
                graf = Figure(figsize=(3, 2))
                g = graf.add_subplot(3, 1, 1)        
                g.plot(ep, loss)
                g.set_title ("Training Loss")
                g.set_ylabel('Loss')
                g.set_xlabel('Epochs')
                g.grid(True)                
                canvas = FigureCanvasTkAgg(graf, master=self.master)
                canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                canvas.draw()
            except :
                pass
        if type_nn.strip() =="Сверточная":
            os.system("signal_cnn.py -t=True -e %i"%n_epoh)
            try:
                loss = pickle.loads(open("train_history_cnn.bin", "rb").read())
                ep = range(1, len(loss)+1)
                graf = Figure(figsize=(3, 2))
                g = graf.add_subplot(3, 1, 1)        
                g.plot(ep, loss)
                g.set_title ("Training Loss")
                g.set_ylabel('Loss')
                g.set_xlabel('Epochs')
                g.grid(True)                
                canvas = FigureCanvasTkAgg(graf, master=self.master)
                canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                canvas.draw()
            except :
                pass            
        if type_nn.strip() =="Прямого распостранения":
            os.system("signal_ffnn.py -t=True -e %i"%n_epoh)
            try:
                loss = pickle.loads(open("train_history_ffnn.bin", "rb").read())
                ep = range(1, len(loss)+1)
                graf = Figure(figsize=(3, 2))
                g = graf.add_subplot(3, 1, 1)        
                g.plot(ep, loss)
                g.set_title ("Training Loss")
                g.set_ylabel('Loss')
                g.set_xlabel('Epochs')
                g.grid(True)                
                canvas = FigureCanvasTkAgg(graf, master=self.master)
                canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                canvas.draw()
            except :
                pass
           

        def show (self):
            global N_prntbits
            global db
            global type_nn
            global n_epoh
        last_type_nn = type_nn        
        last_db = db
        last_epoh = n_epoh
        try:
            type_nn=self.combobox_select_type_nn.get()
            print("Выбранный тип сети: ", type_nn)
        except :
            print('Ошибка при выборе типа сети')
        try:
            b1=int(self.bit1.get())
            b2=int(self.bit2.get())
            b3=int(self.bit3.get())
            if((b1==0 or b1==1) and
               (b2==0 or b2==1) and
               (b3==0 or b3==1)):
                researh_signal[0]=b1
                researh_signal[1]=b2
                researh_signal[2]=b3
                print('Битовая комбинация сигнала: ', researh_signal)
            else:
                print('Используется стандартная битовая комбинация сигнала: ', researh_signal)
        except :
            print('Ошибка при вводе битовой комбинации')
            print('Используется стандартная битовая комбинация сигнала: ', researh_signal)

        try:
           db=int(self.noise_db.get())
           print('Установленный шум: ',db)
        except :
            print('Ошибка при вводе шума')
          
        try:
           n_epoh=int(self.epoh.get())
           print('Установленное количества эпох: ',n_epoh)
        except :
            print('Ошибка при вводе количества эпох')

        arr = []
        for bit in researh_signal:
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
        a_n=A/(10*(db/20))
        noise = np.random.uniform(0,2*a_n,600)

        y = np.zeros(0)  
        y = A * np.cos(2 * np.pi * np.multiply(m, t)+noise)

        y2 = np.zeros(0)  
        y2 = A * np.cos(2 * np.pi * np.multiply(m, t))

        self.flabelwon = tk.Label(text=','.join(str(x) for x in researh_signal)+" без шума")
        self.flabelwon.grid(row=0, column=3)

        self.flabel = tk.Label(text=','.join(str(x) for x in researh_signal)+" c шумом")
        self.flabel.grid(row=0, column=1)

        self.master.state('zoomed')
        N_FFT = float(len(y))
        f = np.arange(0, Fs / 2, Fs / N_FFT)  
        y_f =abs( np.fft.fft(y))

        import numpy as np
        import matplotlib . pyplot as plt
        t = np . arange ( -1.0 , 1.0 , 0.001)
        x = np . sin (2* np . pi *t )
        y = np . cos (5* np . pi *t )
        plt . figure (1 , figsize = (8 ,3))
        plt . plot (t , x)
        plt . figure (2)
        plt . plot (t , y)
        plt . ylabel ( ' cos (t) ')
        plt . figure (1)
        plt . ylabel ( ' sin (t) ')

        fig = Figure(figsize=(6, 7))
        a = fig.add_subplot(3, 1, 1)        
        a.plot(t[0: int(Fs * N_prntbits / Fbit)], m[0: (Fs * N_prntbits // Fbit)])
        a.set_title ("Original VCO output versus time")
        a.set_ylabel('Frequency (Hz)')
        a.set_xlabel('Time (s)')        
        a.grid(True)
        b = fig.add_subplot(3, 1, 2)        
        b.plot(t[0:int(Fs * N_prntbits / Fbit)], y[0: int(Fs * N_prntbits / Fbit)], linewidth=1)
        b.set_xlabel('Time (s)')
        b.set_ylabel('Amplitude (V)')
        b.set_title('Amplitude of carrier versus time')
        b.grid(True)
        c = fig.add_subplot(3, 1, 3)        
        c.plot(f[0: int(Fc + Fdev * 2 * N_FFT / Fs)], y_f[0:int(Fc + Fdev * 2 * N_FFT / Fs)])
        c.set_xlabel('Frequency (Hz)')
        c.set_ylabel('Amplitude (dB)')
        c.set_title('Spectrum')
        c.grid(True)
        fig.set_tight_layout(True)
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().grid(row=1, column=1,rowspan=6, sticky=tk.N+tk.S+tk.E+tk.W)
        canvas.draw()

        N_FFT = float(len(y2))
        f = np.arange(0, Fs / 2, Fs / N_FFT)  
        y_f =abs( np.fft.fft(y2))

        figs = Figure(figsize=(6, 7))
        a = figs.add_subplot(3, 1, 1)        
        a.plot(t[0: int(Fs * N_prntbits / Fbit)],m[0: (Fs * N_prntbits // Fbit)])
        a.set_title ("Original VCO output versus time")
        a.set_ylabel('Frequency (Hz)')
        a.set_xlabel('Time (s)')        
        a.grid(True)
        b = figs.add_subplot(3, 1, 2)        
        b.plot(t[0:int(Fs * N_prntbits / Fbit)], y2[0: int(Fs * N_prntbits / Fbit)], linewidth=1)
        b.set_xlabel('Time (s)')
        b.set_ylabel('Amplitude (V)')
        b.set_title('Amplitude of carrier versus time')
        b.grid(True)
        c = figs.add_subplot(3, 1, 3)        
        c.plot(f[0: int(Fc + Fdev * 2 * N_FFT / Fs)], y_f[0:int(Fc + Fdev * 2 * N_FFT / Fs)])
        c.set_xlabel('Frequency (Hz)')
        c.set_ylabel('Amplitude (dB)')
        c.set_title('Spectrum')
        c.grid(True)
        figs.set_tight_layout(True)
        scanvas = FigureCanvasTkAgg(figs, master=self.master)
        scanvas.get_tk_widget().grid(row=1, column=3,rowspan=6, sticky=tk.N+tk.S+tk.E+tk.W)
        scanvas.draw()
        
        if (last_type_nn != type_nn or            
            last_db != db or
            last_epoh != n_epoh):
            self.train()
        else:        
            if type_nn.strip() =="Рекурентная":            
                try:
                    loss = pickle.loads(open("train_history_rnn.bin", "rb").read())
                    ep = range(1, len(loss)+1)
                    graf = Figure(figsize=(3, 2))
                    g = graf.add_subplot(3, 1, 1)        
                    g.plot(ep, loss)
                    g.set_title ("Training Loss")
                    g.set_ylabel('Loss')
                    g.set_xlabel('Epochs')
                    g.grid(True)                
                    canvas = FigureCanvasTkAgg(graf, master=self.master)
                    canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                    canvas.draw()
                except :
                    pass

            if type_nn.strip() =="Сверточная":            
                try:
                    loss = pickle.loads(open("train_history_cnn.bin", "rb").read())
                    ep = range(1, len(loss)+1)
                    graf = Figure(figsize=(3, 2))
                    g = graf.add_subplot(3, 1, 1)        
                    g.plot(ep, loss)
                    g.set_title ("Training Loss")
                    g.set_ylabel('Loss')
                    g.set_xlabel('Epochs')
                    g.grid(True)                
                    canvas = FigureCanvasTkAgg(graf, master=self.master)
                    canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                    canvas.draw()
                except :
                    pass            
            if type_nn.strip() =="Прямого распостранения":            
                try:
                    loss = pickle.loads(open("train_history_ffnn.bin", "rb").read())
                    ep = range(1, len(loss)+1)
                    graf = Figure(figsize=(3, 2))
                    g = graf.add_subplot(3, 1, 1)        
                    g.plot(ep, loss)
                    g.set_title ("Training Loss")
                    g.set_ylabel('Loss')
                    g.set_xlabel('Epochs')
                    g.grid(True)                
                    canvas = FigureCanvasTkAgg(graf, master=self.master)
                    canvas.get_tk_widget().grid(row=6, column=2, sticky=tk.N+tk.S+tk.E+tk.W)
                    canvas.draw()
                except :
                    pass  
        #self.button["state"] = "disabled"
        
        

    

#Генерация сигнала input_ds - массив трех бит [b,b,b], если plot True то вывод при помощи matplotlib.pylab
def generate_signal(input_ds, plot):

    if plot:
        root = tk.Tk()
        root.title("Signal")
        #root.geometry("250x30")
        #root.resizable(0,0)
        w = root.winfo_reqwidth()
        h = root.winfo_reqheight()
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        xr = (ws/2) - (w/2)
        yr = (hs/2) - (h/2)
        root.geometry('+%d+%d' % (xr, yr))
        app = Mwindow(root)
        root.mainloop()

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
    n=A*10**(db/10)
    noise = np.random.uniform(0,n,600)

    y = np.zeros(0)  
    y = A * np.cos(2 * np.pi * np.multiply(m, t)+noise)
   
    return y

# функция для генерации базы данных сигналов(m_signal - массив сигналов, num_signals - количество генереруемых сигналов каждого типа)
def create_train_base():
    # Устанавливаем колчество сигналов каждого типа для обучения
    num_signals=3000
    print('-'*100)
    print("Генерация базы данных сигналов")
    print("Количество сигналов каждого типа ", num_signals)
    print('-'*100)
    # соединение с базой данных, если нет то создаем ее
    connection=None
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
    except :
        f = open (db_path, "w +")
        f.close()
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

    # удаляем старые записи, если БД некоректна создаем таблицу signals
    try:
        sql = "DELETE FROM signals"
        cursor.execute(sql)
        connection.commit()
    except:        
        sql = """
        CREATE TABLE "signals" (
	    "id" INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
	    "data" BLOB,
	    "type"	INTEGER   
        )
        """
        cursor.execute(sql)
        connection.commit()    

    try:
        i=0
        for sg in m_signals:
            print("Создание сигналов типа ", sg)
            # создаем цикл для генерации сигналов
            for r in range(0,num_signals):                
                # гененрируем сигнал и переводим в байты чтоб сохранить в BLOB объекте БД
                result = generate_signal(sg,False).tobytes()                
                cursor.execute("INSERT INTO signals (data, type)  VALUES (? , ?)",(result, i))                
            connection.commit()
            i=i+1
            print("Успешно")

    except :
        print (sys.exc_info()) # Выводим сообщение о ошибке
        
    # Закрываем соединение с БД, если оно существует
    if(connection):
        connection.close()
    print('-'*100)
    print("Генерация завершена успешно")


def main():
    usage = "Использование: %prog [опции] арг"
    parser = OptionParser(usage)
    parser.add_option("-b", "--base",default=False, dest="create_base",
                      help="Создание базы сигналов для обучения нейронной сети. По умолчанию: False")
    
    (options, args) = parser.parse_args()

    if options.create_base:
        create_train_base()
        msvcrt.getch()
    else:
        generate_signal(researh_signal,True)

if __name__ == "__main__":
    main()
