import tkinter as tk
import numpy as np
import pickle
from classes.Network import Network


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.net = None
        self.canvas_reduce = None
        self.btnPredict = None
        self.btnClear = None
        self.lblNumber = None
        self.panel = None
        self.canvas = None
        self.height_canvas = None
        self.width_canvas = None
        self.array_canvas = None
        self.array_canvas_escaled = None
        self.window_width = None
        self.window_height = None

        self.title('AI Number Detector')
        self.set_window(550, 340)
        self.create_widgets()
        self.setCanvas_logic()
        self.setNetwork_AI()

    def create_widgets(self):
        # Crear un lienzo
        self.width_canvas = 280
        self.height_canvas = 280
        self.canvas = tk.Canvas(self, width=self.width_canvas, height=self.height_canvas, bg='white')
        self.canvas.pack(side=tk.LEFT, padx=30, pady=30)
        self.canvas.bind("<B1-Motion>", self.draw_square)

        # Crear un frame como contenedor para los botones y etiqueta
        self.panel = tk.Frame(self, bg='navy', bd=3, relief='raised')
        self.panel.pack(side=tk.LEFT, padx=30, pady=30, fill=tk.BOTH, expand=True)

        # Crear una etiqueta
        self.lblNumber = tk.Label(self.panel, text=" ", font=("Consolas", 20))
        self.lblNumber.pack(side=tk.BOTTOM, pady=10, padx=10)

        # Crear dos botones
        self.btnClear = tk.Button(self.panel, text="CLEAR", command=self.clear_canvas)
        self.btnClear.pack(side=tk.TOP, pady=10, padx=10)

        self.btnPredict = tk.Button(self.panel, text="PREDICT", command=self.predict)
        self.btnPredict.pack(side=tk.BOTTOM, pady=10, padx=10)

    def draw_square(self, event):
        side = 16
        x1, y1 = int(event.x - side / 2), int(event.y - side / 2)
        x2, y2 = int(event.x + side / 2), int(event.y + side / 2)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
        # Actualizar el arreglo bidimensional
        for i in range(max(0, y1), min(self.height_canvas, y2)):
            for j in range(max(0, x1), min(self.width_canvas, x2)):
                self.array_canvas[i][j] = 1
        # Actualizar el arreglo bidimensional escalado
        r = self.canvas_reduce = 10
        for i, a in enumerate(self.array_canvas_escaled):
            for j, b in enumerate(a):
                self.array_canvas_escaled[i][j] = np.mean(self.array_canvas[i*r:(i*r)+r, j*r:(j*r)+r])

    def predict(self):
        array_num = np.reshape(self.array_canvas_escaled, (784, 1))
        res = self.net.feedforward(array_num)
        self.lblNumber["text"] = f"{np.argmax(res)}"

    def clear_canvas(self):
        self.canvas.delete("all")
        self.lblNumber["text"] = " "
        self.setCanvas_logic()

    def setNetwork_AI(self):
        self.net = Network([784, 30, 10])

        with open('data/weights.pkl', 'rb') as w:
            self.net.weights = pickle.load(w)
        with open('data/biases.pkl', 'rb') as b:
            self.net.biases = pickle.load(b)

    def setCanvas_logic(self):
        self.array_canvas = np.array([[0 for _ in range(self.width_canvas)] for _ in range(self.height_canvas)], dtype=np.float32)
        self.canvas_reduce = 10
        self.array_canvas_escaled = np.array([[0 for _ in range(self.width_canvas//self.canvas_reduce)] for _ in range(self.height_canvas//self.canvas_reduce)], dtype=np.float32)

    def set_window(self, width, height):
        self.window_width = width
        self.window_height = height

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        center_x = int(screen_width / 2 - self.window_width / 2)
        center_y = int(screen_height / 2 - self.window_height / 2)
        self.geometry(f'{self.window_width}x{self.window_height}+{center_x}-{center_y}')
        self.resizable(False, False)


if __name__ == '__main__':
    app = App()
    app.mainloop()
