import tkinter as tk
from PIL import Image, ImageTk

def show_pixel_coordinates(event):
    x, y = event.x, event.y
    pixel_value = image.getpixel((x, y))
    pixel_coordinates_label.config(text=f"Pixel Coordinates: ({x}, {y})  Pixel Value: {pixel_value}")
    print(f"({x}, {y})")
    pixel_coordinates_label.place(x=event.x, y=event.y)

root = tk.Tk()
root.title("Pixel Coordinates Viewer")
root.attributes('-fullscreen', True)

image = Image.open(r"C:\Users\asyao\PycharmProjects\FD\assets\hairline3.png")
tk_image = ImageTk.PhotoImage(image)

label = tk.Label(root, image=tk_image)
label.pack()

pixel_coordinates_label = tk.Label(root, text="", padx=10, pady=10)
pixel_coordinates_label.pack()

label.bind("<Button-1>", show_pixel_coordinates)

root.mainloop()
