
import tkinter as tk
from tkinter import Text, filedialog, Label, Button, Scale, Toplevel, messagebox
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tempfile
import os

class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")
        self.master.geometry("600x500")

        # Створення вкладок для різних функцій
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both')

        # Панель для управління класами
        self.class_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.class_frame, text="Class Management")

        # Панель для невідомих зображень
        self.unknown_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.unknown_frame, text="Unknown Image")

        # Віджети на панелі класів
        self.create_class_widgets()

        # Віджети для завантаження невідомих зображень та класифікації
        self.create_unknown_widgets()

        # Зберігаємо зображення для кожного класу
        self.class_images = {'A': [], 'B': [], 'C': []}
        self.class_vectors = {'A': [], 'B': [], 'C': []}

        # Вікна для зображень кожного класу
        self.create_class_windows()

    def select_crop_area(self, filepath):
        """ Дозволяє користувачу вибрати область для обрізки зображення. """
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        # Використовуємо OpenCV для вибору області
        r = cv2.selectROI("Select Crop Area", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        # Обрізаємо зображення до вибраної області
        cropped_img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        if cropped_img.size == 0:
            print("Error: Crop area is empty.")
            return None

        # Повертаємо обрізане зображення
        return cropped_img

    def create_class_widgets(self):
        """ Створення віджетів для управління класами. """
        # Слайдери для параметрів сегментації
        self.threshold_slider = Scale(self.class_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.grid(row=0, column=0, padx=10, pady=10)

        self.segments_slider = Scale(self.class_frame, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Segments")
        self.segments_slider.grid(row=0, column=1, padx=10, pady=10)

        # Кнопки для завантаження зображень
        self.upload_class_a_button = Button(self.class_frame, text="Upload Class A Images", command=lambda: self.upload_images('A'))
        self.upload_class_a_button.grid(row=1, column=0, padx=10, pady=10)

        self.upload_class_b_button = Button(self.class_frame, text="Upload Class B Images", command=lambda: self.upload_images('B'))
        self.upload_class_b_button.grid(row=1, column=1, padx=10, pady=10)

        self.upload_class_c_button = Button(self.class_frame, text="Upload Class C Images", command=lambda: self.upload_images('C'))
        self.upload_class_c_button.grid(row=1, column=2, padx=10, pady=10)

        # Текстові блоки для векторів
        self.vector_text_a = Text(self.class_frame, height=5, width=40)
        self.vector_text_a.grid(row=2, column=0, padx=10, pady=10)

        self.vector_text_b = Text(self.class_frame, height=5, width=40)
        self.vector_text_b.grid(row=2, column=1, padx=10, pady=10)

        self.vector_text_c = Text(self.class_frame, height=5, width=40)
        self.vector_text_c.grid(row=2, column=2, padx=10, pady=10)

    def create_unknown_widgets(self):
        """ Створення віджетів для роботи з невідомими зображеннями. """
        # Кнопка для завантаження невідомого зображення
        self.upload_unknown_button = Button(self.unknown_frame, text="Upload Unknown Image", command=self.upload_unknown_image)
        self.upload_unknown_button.pack(padx=10, pady=10)

        # Кнопка для класифікації зображення
        self.classify_button = Button(self.unknown_frame, text="Classify Unknown Image", command=self.classify_image)
        self.classify_button.pack(padx=10, pady=10)

        # Текстовий блок для вектора невідомого зображення
        self.unknown_vector_text = Text(self.unknown_frame, height=10, width=60)
        self.unknown_vector_text.pack(padx=10, pady=10)

    def create_class_windows(self):
        """ Створюємо вікна для відображення зображень кожного класу. """
        self.class_windows = {}
        self.class_images_refs = {}  # Словник для зберігання посилань на зображення
        for class_name in ['A', 'B', 'C']:
            self.class_windows[class_name] = Toplevel(self.master)
            self.class_windows[class_name].title(f"Class {class_name} Images")
            self.class_windows[class_name].geometry("400x400")
            self.class_images_refs[class_name] = []  # Ініціалізація списку для зберігання посилань на зображення

    def upload_images(self, class_name):
        """ Завантажуємо зображення для вказаного класу та обрізаємо їх. """
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if filepaths:
            for filepath in filepaths:
                cropped_img = self.select_crop_area(filepath)
                if cropped_img is not None:
                    # Зберігаємо обрізане зображення у тимчасовий файл
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    cv2.imwrite(temp_file.name, cropped_img)
                    self.class_images[class_name].append(temp_file.name)
            self.process_class_images(class_name)
            self.display_class_images(class_name)

    def process_class_images(self, class_name):
        """ Обробляємо завантажені зображення для класу. """
        vectors = []
        for filepath in self.class_images[class_name]:
            vector = self.process_image(filepath)
            if vector:
                vectors.append(vector)

        # Зберігаємо вектори для класу
        self.class_vectors[class_name] = vectors

        # Обчислюємо середні значення для вектора центроїда
        centroid = self.calculate_centroid(vectors)

        # Форматуємо вектори з заголовками для красивого виведення
        formatted_vectors = ""
        for i, vector in enumerate(vectors):
            formatted_vectors += f"Absolute Vector: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Shkliar S{i + 1}: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Shkliar M{i + 1}: [{', '.join([f'{max(val, 0.99):.2f}' for val in vector])}]\n\n"
            formatted_vectors += f"Shkliar S{i + 1}Centr: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Shkliar M{i + 1}Centr (середній вектор): [{', '.join([f'{val:.2f}' for val in centroid])}]\n\n"

        # Оновлюємо текстові блоки для векторів
        if class_name == 'A':
            self.vector_text_a.insert(tk.END, formatted_vectors)
        elif class_name == 'B':
             self.vector_text_b.insert(tk.END, formatted_vectors)
        elif class_name == 'C':
            self.vector_text_c.insert(tk.END, formatted_vectors)


    def display_class_images(self, class_name):
        """ Виводимо зображення для певного класу у відповідному вікні. """
        window = self.class_windows[class_name]

        # Очищаємо вікно від попередніх зображень
        for widget in window.winfo_children():
            widget.destroy()

        # Очищаємо список зображень
        self.class_images_refs[class_name] = []

        # Виводимо всі зображення
        for filepath in self.class_images[class_name]:
            img_with_segments = self.draw_segments(filepath)
            if img_with_segments:
                tk_image = ImageTk.PhotoImage(img_with_segments)

                # Створюємо мітку для зображення та додаємо її у вікно
                image_label = Label(window, image=tk_image)
                image_label.pack()
                self.class_images_refs[class_name].append(
                    tk_image)  # Зберігаємо посилання на зображення, щоб уникнути видалення

    def draw_segments(self, filepath):
        """ Малюємо сегменти на зображенні для візуалізації. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        # Кількість сегментів
        segments = self.segments_slider.get()
        height, width = img.shape
        segment_width = width // segments

        # Малюємо вертикальні лінії для кожного сегменту
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, segments):
            cv2.line(img_color, (i * segment_width, 0), (i * segment_width, height), (255, 0, 0), 1)

        # Перетворюємо на зображення для tkinter
        img_pil = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((100, 100))  # Resize for display
        return img_pil

    def process_image(self, filepath):
        """ Обробка зображення, створення векторів ознак із сегментацією. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        # Порогове перетворення
        _, thresholded = cv2.threshold(img, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)

        # Сегментація та формування векторів
        segments = self.segments_slider.get()
        height, width = thresholded.shape
        segment_width = width // segments

        absolute_vector = []
        for i in range(segments):
            segment = thresholded[:, i * segment_width:(i + 1) * segment_width]
            count_black_pixels = np.sum(segment == 0)
            absolute_vector.append(count_black_pixels)

        # Нормалізація вектора
        normalized_vector = [x / sum(absolute_vector) for x in absolute_vector] if sum(absolute_vector) > 0 else [0] * segments

        return normalized_vector

    def calculate_centroid(self, vectors):
        """ Обчислюємо центр мас для вказаних векторів класу. """
        return [sum(col) / len(col) for col in zip(*vectors)]

    def euclidean_distance(self, vector1, vector2):
        """ Обчислення Евклідової відстані між двома векторами. """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))

    def upload_unknown_image(self):
        """ Завантажуємо невідоме зображення і відображаємо його вектор ознак. """
        self.unknown_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.unknown_image_path:
            self.unknown_vector = self.process_image(self.unknown_image_path)
            if self.unknown_vector:
                # Виводимо шлях та вектор невідомого зображення
                self.unknown_vector_text.insert(tk.END, f"Unknown Image Path: {self.unknown_image_path}\n")

                # Формуємо відформатований вектор з обмеженою кількістю знаків після коми
                formatted_vector = ', '.join([f'{val:.2f}' for val in self.unknown_vector])
                self.unknown_vector_text.insert(tk.END, f"Vector: [{formatted_vector}]\n")

    def classify_image(self):
        """ Класифікуємо невідоме зображення на основі відстаней до центрів мас класів. """
        if not self.unknown_vector:
            messagebox.showerror("Error", "Please upload an unknown image.")
            return

        distances = {}

        for class_name in ['A', 'B', 'C']:
            if not self.class_vectors[class_name]:
                continue

            # Обчислюємо центр мас для класу
            centroid = self.calculate_centroid(self.class_vectors[class_name])

            # Обчислюємо Евклідову відстань
            distance = self.euclidean_distance(self.unknown_vector, centroid)
            distances[class_name] = distance

        # Знаходимо клас з мінімальною відстанню
        if distances:
            min_class = min(distances, key=distances.get)
            classification_result = f"Unknown Image belongs to Class {min_class}"
        else:
            classification_result = "Unknown Image does not belong to any class."

        # Виводимо результат класифікації та відстані
        self.unknown_vector_text.insert(tk.END, f"Classification Result: {classification_result}\n")
        for class_name, distance in distances.items():
            self.unknown_vector_text.insert(tk.END, f"Euclidean Distance to Class {class_name}: {distance:.2f}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureExtractionApp(root)
    root.mainloop()
