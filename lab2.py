import tkinter as tk
from tkinter import Text, filedialog, Label, Button, Entry, Toplevel, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")
        self.master.geometry("600x500")

        # Зберігаємо зображення для кожного класу
        self.class_images = {'A': [], 'B': [], 'C': []}
        self.class_vectors = {'A': [], 'B': [], 'C': []}
        self.unknown_vector = []

        # Створюємо вкладки
        tab_control = ttk.Notebook(master)

        self.tab_classes = ttk.Frame(tab_control)
        self.tab_unknown = ttk.Frame(tab_control)
        self.tab_settings = ttk.Frame(tab_control)

        tab_control.add(self.tab_classes, text="Class Images")
        tab_control.add(self.tab_unknown, text="Unknown Image")
        tab_control.add(self.tab_settings, text="Settings")
        tab_control.pack(expand=True, fill='both')

        # Створюємо вікна для зображень кожного класу
        self.create_class_windows()

        # Таб: налаштування
        self.create_settings_tab()

        # Таб: зображення класів
        self.create_class_images_tab()

        # Таб: невідоме зображення
        self.create_unknown_image_tab()

        # Для збереження шляху до невідомого зображення
        self.unknown_image_path = None

        # Прогрес-бар
        self.progress = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

    def create_settings_tab(self):
        """ Створюємо вкладку з налаштуваннями. """
        settings_frame = ttk.LabelFrame(self.tab_settings, text="Settings", padding=(20, 10))
        settings_frame.pack(padx=10, pady=10, fill='x')

        # Поле для введення порогу (threshold)
        self.threshold_label = Label(settings_frame, text="Threshold (0-255):")
        self.threshold_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.threshold_entry = Entry(settings_frame, validate="key")
        self.threshold_entry.grid(row=0, column=1, padx=5, pady=5)
        self.threshold_entry.bind("<KeyRelease>", self.validate_entry)

        # Поле для введення кількості сегментів (segments)
        self.segments_label = Label(settings_frame, text="Number of Segments (2-10):")
        self.segments_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        self.segments_entry = Entry(settings_frame, validate="key")
        self.segments_entry.grid(row=1, column=1, padx=5, pady=5)
        self.segments_entry.bind("<KeyRelease>", self.validate_entry)

    def create_class_images_tab(self):
        """ Створюємо вкладку для зображень класів. """
        # Кнопки для завантаження зображень для кожного класу
        class_frame = ttk.LabelFrame(self.tab_classes, text="Class Images", padding=(20, 10))
        class_frame.pack(padx=10, pady=10, fill='both')

        self.upload_class_a_button = Button(class_frame, text="Upload Class A Images",
                                            command=lambda: self.upload_images('A'), bg="lightblue", fg="white")
        self.upload_class_a_button.grid(row=0, column=0, padx=10, pady=10)

        self.upload_class_b_button = Button(class_frame, text="Upload Class B Images",
                                            command=lambda: self.upload_images('B'), bg="lightgreen", fg="white")
        self.upload_class_b_button.grid(row=0, column=1, padx=10, pady=10)

        self.upload_class_c_button = Button(class_frame, text="Upload Class C Images",
                                            command=lambda: self.upload_images('C'), bg="lightcoral", fg="white")
        self.upload_class_c_button.grid(row=0, column=2, padx=10, pady=10)

        # Текстові блоки для векторів кожного класу
        self.vector_text_a = Text(class_frame, height=10, bg="lightyellow")
        self.vector_text_a.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.vector_text_b = Text(class_frame, height=10, bg="lightyellow")
        self.vector_text_b.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.vector_text_c = Text(class_frame, height=10, bg="lightyellow")
        self.vector_text_c.grid(row=1, column=2, padx=10, pady=10, sticky='nsew')

    def create_unknown_image_tab(self):
        """ Створюємо вкладку для невідомого зображення. """
        unknown_frame = ttk.LabelFrame(self.tab_unknown, text="Unknown Image Classification", padding=(20, 10))
        unknown_frame.pack(padx=10, pady=10, fill='both')

        # Кнопка для завантаження невідомого зображення
        self.upload_unknown_button = Button(unknown_frame, text="Upload Unknown Image", command=self.upload_unknown_image)
        self.upload_unknown_button.grid(row=0, column=0, padx=10, pady=10)

        # Кнопка для класифікації зображення
        self.classify_button = Button(unknown_frame, text="Classify Unknown Image", command=self.classify_image)
        self.classify_button.grid(row=0, column=1, padx=10, pady=10)

        # Текстовий блок для вектора невідомого зображення
        self.unknown_vector_text = Text(unknown_frame, height=10, bg="lightgrey", relief=tk.SUNKEN, bd=3)
        self.unknown_vector_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

    def create_class_windows(self):
        """ Створюємо вікна для відображення зображень кожного класу. """
        self.class_windows = {}
        for class_name in ['A', 'B', 'C']:
            self.class_windows[class_name] = Toplevel(self.master)
            self.class_windows[class_name].title(f"Class {class_name} Images")
            self.class_windows[class_name].geometry("400x400")

    def upload_images(self, class_name):
        """ Завантажуємо зображення для вказаного класу. """
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if filepaths:
            self.class_images[class_name] = list(filepaths)
            self.process_class_images(class_name)

    def process_class_images(self, class_name):
        """ Обробляємо завантажені зображення для класу. """
        vectors = []
        for filepath in self.class_images[class_name]:
            vector = self.process_image(filepath)
            if vector:
                vectors.append(vector)

        self.class_vectors[class_name] = vectors
        formatted_vectors = ""
        for i, vector in enumerate(vectors):
            formatted_vectors += f"Absolute Vector: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Shkliar S{i + 1}: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Shkliar M{i + 1}: [{', '.join([f'{max(val, 0.99):.2f}' for val in vector])}]\n\n"

        if class_name == 'A':
            self.vector_text_a.insert(tk.END, formatted_vectors)
        elif class_name == 'B':
            self.vector_text_b.insert(tk.END, formatted_vectors)
        elif class_name == 'C':
            self.vector_text_c.insert(tk.END, formatted_vectors)

        self.display_class_images(class_name)

    def display_class_images(self, class_name):
        """ Виводимо зображення для певного класу у відповідному вікні. """
        window = self.class_windows[class_name]
        for widget in window.winfo_children():
            widget.destroy()

        for filepath in self.class_images[class_name]:
            img_with_segments = self.draw_segments(filepath)
            tk_image = ImageTk.PhotoImage(img_with_segments)
            image_label = Label(window, image=tk_image)
            image_label.image = tk_image  # Keep reference
            image_label.pack(side=tk.LEFT)

    def draw_segments(self, filepath):
        """ Малюємо сегменти на зображенні для візуалізації. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        try:
            segments = int(self.segments_entry.get())
            if segments < 2 or segments > 10:
                raise ValueError("Number of segments should be between 2 and 10.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid segments value: {str(e)}")
            return None

        height, width = img.shape
        segment_width = width // segments
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, segments):
            cv2.line(img_color, (i * segment_width, 0), (i * segment_width, height), (255, 0, 0), 1)

        img_pil = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((100, 100))
        return img_pil

    def process_image(self, filepath):
        """ Обробка зображення, створення векторів ознак із сегментацією. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        try:
            threshold_value = int(self.threshold_entry.get())
            if threshold_value < 0 or threshold_value > 255:
                raise ValueError("Threshold should be between 0 and 255.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid threshold value: {str(e)}")
            return None

        _, thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
        try:
            segments = int(self.segments_entry.get())
            if segments < 2 or segments > 10:
                raise ValueError("Number of segments should be between 2 and 10.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid segments value: {str(e)}")
            return None

        height, width = thresholded.shape
        segment_width = width // segments
        absolute_vector = []
        for i in range(segments):
            segment = thresholded[:, i * segment_width:(i + 1) * segment_width]
            count_black_pixels = np.sum(segment == 0)
            absolute_vector.append(count_black_pixels)

        normalized_vector = [x / sum(absolute_vector) for x in absolute_vector] if sum(absolute_vector) > 0 else [0] * segments
        return normalized_vector

    def calculate_class_max_min(self, class_name):
        """ Обчислюємо максимальні та мінімальні вектори для кожного класу. """
        vectors = self.class_vectors[class_name]
        if not vectors:
            return None, None

        max_vector = np.max(vectors, axis=0)
        min_vector = np.min(vectors, axis=0)
        return max_vector, min_vector

    def upload_unknown_image(self):
        """ Завантажуємо невідоме зображення. """
        self.unknown_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.unknown_image_path:
            self.unknown_vector = self.process_image(self.unknown_image_path)
            if self.unknown_vector:
                self.unknown_vector_text.delete(1.0, tk.END)
                self.unknown_vector_text.insert(tk.END, f"Path: {self.unknown_image_path}\n")
                self.unknown_vector_text.insert(tk.END, f"Vector: [{', '.join(map(lambda x: f'{x:.5f}', self.unknown_vector))}]\n")

    def classify_image(self):
        """ Класифікуємо невідоме зображення на основі максимальних та мінімальних векторів кожного класу. """
        if not self.unknown_vector:
            messagebox.showerror("Error", "Please upload an unknown image.")
            return

        tolerance = 0.05  # Допуск для порівняння векторів
        classification_result = None

        for class_name in ['A', 'B', 'C']:
            max_vector, min_vector = self.calculate_class_max_min(class_name)
            if max_vector is None or min_vector is None:
                continue

            in_range = all(min_val - tolerance <= val <= max_val + tolerance for val, min_val, max_val in
                           zip(self.unknown_vector, min_vector, max_vector))

            if in_range:
                classification_result = f"Unknown Image belongs to Class {class_name}"
                break

        if classification_result is None:
            classification_result = "Unknown Image does not belong to any class."

        self.unknown_vector_text.insert(tk.END, f"{classification_result}\n")

    def validate_entry(self, event):
        """ Перевірка введених значень у полях налаштувань у реальному часі. """
        try:
            threshold_value = int(self.threshold_entry.get())
            segments_value = int(self.segments_entry.get())
            if threshold_value < 0 or threshold_value > 255 or segments_value < 2 or segments_value > 10:
                raise ValueError
            self.progress['value'] = 100  # Оновлюємо прогрес
        except ValueError:
            self.progress['value'] = 0  # Якщо значення некоректні, прогрес - 0


root = tk.Tk()
app = FeatureExtractionApp(root)
root.mainloop()
