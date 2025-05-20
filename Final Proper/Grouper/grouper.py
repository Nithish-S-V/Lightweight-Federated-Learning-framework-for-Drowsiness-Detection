import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ----------------------- Configuration -----------------------
dataset_folder = "drowsy"  # Folder where images are stored.

# Define the class names
class_names = ["drowsy", "notdrowsy", "closed", "open", "yawn", "notyawn"]
delete_option = "delete"

# Create class folders if they do not exist
for cls in class_names:
    os.makedirs(os.path.join(dataset_folder, cls), exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(dataset_folder) 
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

current_index = 0  # Tracks the current image

# ------------------- Function Definitions -------------------

def load_next_image():
    """
    Loads and displays the next image.
    """
    global current_index, image_files
    if current_index < len(image_files):
        img_path = os.path.join(dataset_folder, image_files[current_index])
        img = Image.open(img_path)
        img = img.resize((800, 800))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        filename_label.config(text=image_files[current_index])
        reset_checkboxes()
    else:
        filename_label.config(text="No more images")
        panel.config(image="")

def reset_checkboxes():
    """
    Unchecks all checkboxes.
    """
    for var in checkbox_vars:
        var.set(0)

def classify_image():
    """
    Moves the image to the selected class folders.
    """
    global current_index, image_files
    if current_index < len(image_files):
        selected_classes = [class_names[i] for i, var in enumerate(checkbox_vars) if var.get() == 1]
        if not selected_classes:
            messagebox.showwarning("No Selection", "Please select at least one class.")
            return

        current_file = image_files[current_index]
        src_path = os.path.join(dataset_folder, current_file)

        for cls in selected_classes:
            dest_path = os.path.join(dataset_folder, cls, current_file)
            shutil.copy(src_path, dest_path)  # Copy instead of move

        os.remove(src_path)  # Remove original file after classification
        current_index += 1
        load_next_image()

def on_delete():
    """
    Deletes the current image.
    """
    global current_index, image_files
    if current_index < len(image_files):
        current_file = image_files[current_index]
        os.remove(os.path.join(dataset_folder, current_file))
        current_index += 1
        load_next_image()

def on_skip():
    """
    Skips the current image without making any changes.
    """
    global current_index
    current_index += 1
    load_next_image()

# ----------------------- Build the GUI -----------------------
root = tk.Tk()
root.title("Image Sorting Tool")
root.configure(bg="#121212")  # Dark background
root.geometry("1920x1080")

# Styling options
btn_bg = "#333"  # Button background
btn_fg = "white"  # Button text color
btn_hover = "#555"
frame_bg = "#1e1e1e"

# Button Animation
def on_enter(e):
    e.widget.config(bg=btn_hover)

def on_leave(e):
    e.widget.config(bg=btn_bg)

# Main Layout Frames
main_frame = tk.Frame(root, bg="#121212")
main_frame.pack(fill=tk.BOTH, expand=True)

# Left Panel (Image Display)
left_panel = tk.Frame(main_frame, bg=frame_bg, relief="raised", bd=2, width=960, height=1080)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

filename_label = tk.Label(left_panel, text="", font=("Arial", 12), fg="white", bg=frame_bg)
filename_label.pack()

panel = tk.Label(left_panel, bg=frame_bg)
panel.pack(pady=5)

# Right Panel (Buttons)
right_panel = tk.Frame(main_frame, bg="#121212", width=960, height=1080)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Checkbox Frame
checkbox_frame = tk.Frame(right_panel, bg="#121212")
checkbox_frame.pack(pady=30)

checkbox_vars = []
for i, cls in enumerate(class_names):
    var = tk.IntVar()
    checkbox_vars.append(var)
    chk = tk.Checkbutton(checkbox_frame, text=cls.upper(), variable=var, font=("Arial", 14, "bold"),
                         bg="#121212", fg="white", selectcolor="#333", activebackground="#555")
    chk.grid(row=i // 2, column=i % 2, padx=40, pady=10)

# Action Buttons
button_frame = tk.Frame(right_panel, bg="#121212")
button_frame.pack(pady=30)

submit_btn = tk.Button(button_frame, text="SUBMIT", command=classify_image, font=("Arial", 14, "bold"),
                        bg=btn_bg, fg=btn_fg, activebackground=btn_hover, width=15, height=2, relief="flat")
submit_btn.grid(row=0, column=0, padx=40, pady=20)
submit_btn.bind("<Enter>", on_enter)
submit_btn.bind("<Leave>", on_leave)

delete_btn = tk.Button(button_frame, text="DELETE", command=on_delete, font=("Arial", 14, "bold"),
                        bg="red", fg="white", width=15, height=2, relief="flat")
delete_btn.grid(row=0, column=1, padx=40, pady=20)

skip_btn = tk.Button(button_frame, text="SKIP", command=on_skip, font=("Arial", 14, "bold"),
                     bg="#666", fg="white", width=15, height=2, relief="flat")
skip_btn.grid(row=1, column=0, columnspan=2, padx=40, pady=20)

load_next_image()
root.mainloop()