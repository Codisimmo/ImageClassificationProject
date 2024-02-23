import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab, Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    if file_path:
        global image_for_canvas
        loaded_image = Image.open(file_path)
        loaded_image.thumbnail((400, 400))  # přizpůsobení velikosti obrázku
        image_for_canvas = ImageTk.PhotoImage(loaded_image)
        canvas.create_image(200, 200, image=image_for_canvas)

def load_category_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]
category_names = load_category_names('categories.txt')

def draw(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=1)   

def clear_canvas():
    canvas.delete('all')

def vyhodnotit():
    
    # Získání souřadnic a uložení .png
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    filename = "cmaranice.png"
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # Načtení obrázku a příprava pro model
    img = image.load_img(filename, target_size=(70, 70))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Klasifikace obrázku
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    max_probability = np.max(predictions)
    predicted_category_name = category_names[predicted_class[0]] 

    #vypsání výsledku různými způsoby
    category_probabilities = [(category_names[i], prob) for i, prob in enumerate(predictions[0])]
    sorted_category_probabilities = sorted(category_probabilities, key=lambda x: x[1], reverse=True)
    for category, probability in sorted_category_probabilities:
        print(f"{category}: {probability}")
 
    threshold1 = 0.5
    threshold2 = 0.8
    if max_probability < threshold1:
        print("Toto nepoznávám.")
    elif threshold1 <= max_probability < threshold2:
        print(f"Nejsem si jistý, možná {predicted_category_name}")
    else:
        print(f"Toto je: {predicted_category_name}")

# Vytvoření hlavního okna
root = tk.Tk()
root.title("Poznám tvůj obrázek")

canvas = tk.Canvas(root, width=400, height=400, bg='white')
canvas.pack()
canvas.bind('<B1-Motion>', draw)

clear_button = tk.Button(root, text="Vymazat", command=clear_canvas)
clear_button.pack() 

save_button = tk.Button(root, text="Vyhodnotit", command=vyhodnotit)
save_button.pack()

open_button = tk.Button(root, text="Otevřít soubor", command=open_file)
open_button.pack()

root.mainloop()