from __future__ import annotations

import argparse
import glob
import json
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

from PIL import Image
from PIL import ImageTk

parser = argparse.ArgumentParser(description='Run the Image Labeler App')
parser.add_argument(
    '--auto', type=str, default='no', choices=['yes', 'no'],
    help='Automatically select directories (yes or no)',
)
args = parser.parse_args()
auto = args.auto == 'yes'


class ImageLabelerApp:
    def __init__(self, master, img_directory, data_file, sources_directory, properties_directory):
        self.master = master
        self.img_directory = img_directory
        self.sources_directory = sources_directory
        self.properties_directory = properties_directory
        self.data_file = data_file
        self.master.title('ICs selector')
        self.master.configure(background='black')

        self.data = self.load_data()

        # 加载图片
        self.images = [
            f for f in os.listdir(
                img_directory,
            ) if f.endswith(('png', 'jpg', 'jpeg'))
        ]
        self.images.sort()
        self.all_images = [os.path.splitext(f)[0] for f in self.images]
        self.current_index = len(self.data) if self.all_images else -1

        # 设置UI
        self.setup_ui()

        # 显示初始化图片
        if self.current_index != -1:
            self.show_image(self.all_images[self.current_index])

        self.update_progress_label()
        self.master.bind('<Return>', lambda event: self.save_label_and_next())
        self.master.bind('<B>', lambda event: self.go_to_previous_image())

        # New key bindings for arrow keys
        self.master.bind('<Up>', lambda event: self.show_properties_image())
        self.master.bind('<Down>', lambda event: self.show_source_image())
        self.master.bind('<Left>', lambda event: self.go_to_previous_image())
        self.master.bind('<Right>', lambda event: self.save_label_and_next())

    def setup_ui(self):
        self.frame = ttk.Frame(self.master, padding='10 10 10 10')
        self.frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self.frame)
        self.image_label.grid(
            column=1, row=1, columnspan=2, sticky=(tk.W, tk.E),
        )

        self.filename_label = ttk.Label(
            self.frame, text='', background='light gray',
        )
        self.filename_label.grid(
            column=1, row=2, columnspan=2, sticky=(tk.W, tk.E),
        )

        self.text_entry = ttk.Entry(self.frame, width=20)
        self.text_entry.grid(column=1, row=3, columnspan=2, sticky=tk.EW)

        self.progress_label = ttk.Label(
            self.frame, text='', background='light gray',
        )
        self.progress_label.grid(
            column=1, row=4, columnspan=2, sticky=(tk.W, tk.E),
        )

        self.next_button = ttk.Button(
            self.frame, text='Next', command=self.save_label_and_next,
        )
        self.next_button.grid(column=2, row=5, sticky=tk.EW)

        self.prev_button = ttk.Button(
            self.frame, text='Previous', command=self.go_to_previous_image,
        )
        self.prev_button.grid(column=1, row=5, sticky=tk.EW)

        self.sources_button = ttk.Button(
            self.frame, text='Sources', command=self.show_source_image,
        )
        self.sources_button.grid(column=1, row=6, sticky=tk.EW)

        self.properties_button = ttk.Button(
            self.frame, text='Properties', command=self.show_properties_image,
        )
        self.properties_button.grid(column=2, row=6, sticky=tk.EW)

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file) as file:
                return json.load(file)
        else:
            with open(self.data_file, 'w') as file:
                json.dump({}, file)
            return {}

    def show_image(self, filename):
        path = os.path.join(self.img_directory, filename + '.png')
        img = Image.open(path)
        base_height = 750
        h_percent = (base_height / float(img.size[1]))
        w_size = int(float(img.size[0]) * float(h_percent))
        img = img.resize((w_size, base_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)
        self.current_image = filename
        self.filename_label.config(
            text=f"Current: {filename}", anchor='center',
        )
        self.update_progress_label()
        self.text_entry.delete(0, tk.END)
        if filename in self.data:
            self.text_entry.insert(0, ','.join(map(str, self.data[filename])))

    def show_source_image(self):
        # Find all source images matching the pattern
        pattern = os.path.join(
            self.sources_directory,
            f"{self.current_image}*.png",
        )
        source_files = sorted(glob.glob(pattern))
        if source_files:
            # Create a new Toplevel window
            source_window = tk.Toplevel(self.master)
            source_window.title(f"Sources for {self.current_image}")
            source_window.configure(background='black')
            # Bind 'Esc' key to close the window
            source_window.bind('<Escape>', lambda e: source_window.destroy())

            images = []
            image_widths = []

            for file in source_files:
                img = Image.open(file)
                # Optionally resize images
                base_height = 600  # Set the desired height
                h_percent = (base_height / float(img.size[1]))
                w_size = int(float(img.size[0]) * float(h_percent))
                img = img.resize(
                    (w_size, base_height),
                    Image.Resampling.LANCZOS,
                )
                tk_img = ImageTk.PhotoImage(img)
                # Keep a reference to prevent garbage collection
                images.append(tk_img)
                image_widths.append(w_size)

            total_width = sum(image_widths)
            max_height = base_height

            # Create a frame inside the window
            frame = ttk.Frame(source_window)
            frame.pack(fill=tk.BOTH, expand=True)

            for tk_img in images:
                label = ttk.Label(frame, image=tk_img)
                label.pack(side='left', padx=5)

            # Adjust the window size to fit all images
            source_window.update_idletasks()
            window_width = total_width + 10  # Add some padding
            window_height = max_height + 10
            source_window.geometry(f"{window_width}x{window_height}")

            # Keep references to images to prevent garbage collection
            source_window.images = images
        else:
            messagebox.showwarning(
                'Warning', 'No sources image found for this run.',
            )

    # def show_source_image(self):
    #     path = os.path.join(self.sources_directory, self.current_image + '.png')
    #     path0 = os.path.join(self.sources_directory, self.current_image + '-0.png')
    #     path1 = os.path.join(self.sources_directory, self.current_image + '-1.png')
    #     if os.path.exists(path0) and os.path.exists(path1):
    #         img0 = Image.open(path0)
    #         img0.show(title=f'{self.current_image}-sources-1')
    #         img1 = Image.open(path1)
    #         img1.show(title=f'{self.current_image}-sources-2')
    #     elif os.path.exists(path):
    #         img = Image.open(path)
    #         img.show(title=f'{self.current_image}')
    #     else:
    #         messagebox.showwarning("Warning", "No sources image found for this run.")

    def show_properties_image(self):
        i = simpledialog.askinteger(
            'Input', 'Please enter the component number:',
        )
        if i is not None:
            path = os.path.join(
                self.properties_directory,
                self.current_image, f"{i}.png",
            )
            if os.path.exists(path):
                img = Image.open(path)
                base_height = 800  # Set the desired height
                h_percent = (base_height / float(img.size[1]))
                w_size = int(float(img.size[0]) * float(h_percent))
                img = img.resize(
                    (w_size, base_height),
                    Image.Resampling.LANCZOS,
                )
                tk_img = ImageTk.PhotoImage(img)

                # Create a new Toplevel window
                prop_window = tk.Toplevel(self.master)
                prop_window.title(f'{self.current_image}-comp{i}')
                prop_window.configure(background='black')

                # Ensure the properties window has focus
                prop_window.focus_set()

                # Bind 'Esc' key to close the window
                prop_window.bind('<Escape>', lambda e: prop_window.destroy())

                # Create a label to hold the image
                label = ttk.Label(prop_window, image=tk_img)
                label.pack()

                # Keep a reference to the image to prevent garbage collection
                prop_window.image = tk_img
            else:
                messagebox.showwarning(
                    'Warning', 'No properties image found for this component.',
                )

    def update_progress_label(self):
        progress_text = f"[{self.current_index + 1}/{len(self.all_images)}]"
        self.progress_label.config(text=progress_text, anchor='center')

    def save_label_and_next(self):
        label_input = self.text_entry.get()
        if not label_input.strip():
            messagebox.showwarning(
                'Warning', 'Please provide a label before moving to the next image.',
            )
            return
        label_list = [
            int(item) for item in label_input.split(
                ',',
            )
        ] if ',' in label_input else [int(label_input)]
        self.data[self.current_image] = label_list
        with open(self.data_file, 'w') as file:
            json.dump(self.data, file, indent=4)
        self.current_index = (self.current_index + 1) % len(self.all_images)
        if self.current_index == 0:
            messagebox.showinfo(
                'Info', 'All images have been labeled. Now will exiting.',
            )
            self.master.destroy()
        else:
            self.show_image(self.all_images[self.current_index])

    def go_to_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
        else:
            self.current_index = len(self.all_images) - 1
        self.show_image(self.all_images[self.current_index])

    def on_closing(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?'):
            self.master.destroy()


if __name__ == '__main__':

    root = tk.Tk()
    script_dir = os.path.dirname(__file__)
    print(script_dir)
    if auto:
        img_directory = os.path.join(script_dir, 'ICs_all')
        sources_directory = os.path.join(script_dir, 'IC_sources')
        properties_directory = os.path.join(script_dir, 'IC_properties')
    else:
        img_directory = filedialog.askdirectory(title='Select ICs Directory')
        sources_directory = filedialog.askdirectory(
            title='Select Sources Directory',
        )
        properties_directory = filedialog.askdirectory(
            title='Select Properties Directory',
        )
    data_file = os.path.join(script_dir, 'Artifact_ICs.json')
    app = ImageLabelerApp(
        root, img_directory, data_file,
        sources_directory, properties_directory,
    )
    root.protocol('WM_DELETE_WINDOW', app.on_closing)
    root.mainloop()
