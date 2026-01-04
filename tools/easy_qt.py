import os
import random
import tkinter as tk
from tkinter import Label, PhotoImage

from PIL import Image, ImageTk


def get_question_and_answer():
    folders = [
        "_game1_assets/data/tumor types/benign",
        "_game1_assets/data/tumor types/malignant",
    ]
    selected_folder = random.choice(folders)
    correct_answer = os.path.basename(selected_folder)
    image_files = os.listdir(selected_folder)
    random_image = random.choice(image_files)
    return correct_answer, os.path.join(selected_folder, random_image)


def create_button(root, text, size, color, color2, command, x, y):
    button = tk.Button(
        root, text=text, command=command, font=("Tahoma", size), fg=color, bg=color2
    )
    button.place(x=x, y=y)
    return button


def create_label(root, text, size, color, x, y):
    label = tk.Label(root, text=text, font=("Tahoma", size), fg=color)
    label.place(x=x, y=y)
    return label


def create_label2(root, text, size, fg_color, bg_color, x, y):
    label = tk.Label(
        root,
        text=text,
        font=("Tahoma", size),
        fg=fg_color,
        bg=bg_color,
        highlightthickness=0,
    )
    label.place(x=x, y=y)
    return label


def create_image_at_center(root, path, center_x, center_y):
    image = PhotoImage(file=path)
    label = Label(root, image=image)
    label.image = image
    label.place(x=center_x - image.width() // 2, y=center_y - image.height() // 2)
    return label


def create_text(root, text, size, color, x, y):
    label = tk.Label(root, text=text, font=("Arial", size), fg=color)
    label.place(x=x, y=y)
    return label


def handle_click(event, start_x, start_y, finish_x, finish_y, command):
    if start_x <= event.x <= finish_x and start_y <= event.y <= finish_y:
        command()


def create_invisible_button(widget, start_x, start_y, finish_x, finish_y, command):
    widget.bind(
        "<Button-1>",
        lambda event: handle_click(
            event, start_x, start_y, finish_x, finish_y, command
        ),
    )


def add_image(root, path, start_x, start_y, end_x, end_y):
    image = tk.PhotoImage(file=path)
    image = image.subsample((end_x - start_x), (end_y - start_y))
    label = tk.Label(root, image=image)
    label.image = image
    label.place(x=start_x, y=start_y)
    return label


def change_background(root, path):
    image = tk.PhotoImage(file=path)
    label = tk.Label(root, image=image)
    label.image = image
    label.place(x=0, y=0)


def create_entry(root, width, x, y):
    entry = tk.Entry(root, width=width)
    entry.place(x=x, y=y)
    return entry


def create_frame(root, width, height, x, y):
    frame = tk.Frame(root, width=width, height=height, bg="white")
    frame.place(x=x, y=y)
    return frame


def show_page(page):
    page.tkraise(),


def clear_screen(root):
    for widget in root.winfo_children():
        widget.destroy()


def create_info_row(parent, title, initial_value, y_pos):
    frame = tk.Frame(parent, bg="white")
    frame.place(x=0, y=y_pos, width=280, height=40)

    title_label = tk.Label(frame, text=title, bg="white", font=("Arial", 12))
    title_label.pack(side=tk.LEFT, padx=10)

    value_label = tk.Label(
        frame, text=initial_value, bg="white", font=("Arial", 12, "bold"), fg="#d32f2f"
    )
    value_label.pack(side=tk.RIGHT, padx=10)

    return {"frame": frame, "title": title_label, "value": value_label}


def create_button_with_image(root, path, x, y, command):
    try:
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Failed to load button image {path}: {e}")
        return None
    btn = tk.Button(root, image=photo, command=command, borderwidth=0)
    btn.image = photo
    btn.place(x=x, y=y)
    return btn


def create_image(root, path, x, y):
    try:
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Failed to load image {path}: {e}")
        return None
    label = tk.Label(root, image=photo)
    label.image = photo
    label.place(x=x, y=y)
    return label
