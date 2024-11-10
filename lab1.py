import tkinter as tk
from tkinter import filedialog, Label, Button, Entry
import cv2
import numpy as np
from PIL import Image, ImageTk


class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")

        # UI components
        self.label = Label(master, text="Upload an Image:")
        self.label.pack()

        self.upload_button = Button(master, text="Choose File", command=self.upload_image)
        self.upload_button.pack()

        # Threshold input field
        self.threshold_label = Label(master, text="Threshold (0-255):")
        self.threshold_label.pack()

        self.threshold_entry = Entry(master)
        self.threshold_entry.pack()

        # Number of vertical segments input field
        self.segments_label = Label(master, text="Number of Vertical Segments (2-10):")
        self.segments_label.pack()

        self.segments_entry = Entry(master)
        self.segments_entry.pack()

        # Crop button
        self.crop_button = Button(master, text="Crop Image", command=self.crop_image)
        self.crop_button.pack()

        # Process button
        self.process_button = Button(master, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.image_label = Label(master)
        self.image_label.pack()

        self.features_label = Label(master, text="Feature Vectors:")
        self.features_label.pack()

        self.cropped_image = None  # To store the cropped image

    def upload_image(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.display_image(self.filepath)

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_image)

    def crop_image(self):
        if not hasattr(self, 'filepath'):
            self.features_label.config(text="Error: Please upload an image first.")
            return

        # Load the image
        img = cv2.imread(self.filepath)

        # Let user select the region of interest (ROI)
        r = cv2.selectROI("Select Region of Interest", img)
        if r == (0, 0, 0, 0):  # No region was selected
            cv2.destroyAllWindows()
            return

        # Crop the image using the selected coordinates
        x, y, w, h = map(int, r)
        self.cropped_image = img[y:y+h, x:x+w]

        # Display the cropped image
        self.display_cropped_image(self.cropped_image)
        cv2.destroyAllWindows()

    def display_cropped_image(self, img_array):
        # Convert to PIL format for Tkinter
        img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        img = img.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_image)

    def process_image(self):
        # Validate and get the threshold value from entry
        try:
            threshold_value = int(self.threshold_entry.get())
            if threshold_value < 0 or threshold_value > 255:
                raise ValueError("Threshold should be between 0 and 255.")
        except ValueError as e:
            self.features_label.config(text=f"Error: {str(e)}")
            return

        # Validate and get the number of vertical segments from entry
        try:
            segments = int(self.segments_entry.get())
            if segments < 2 or segments > 10:
                raise ValueError("Number of segments should be between 2 and 10.")
        except ValueError as e:
            self.features_label.config(text=f"Error: {str(e)}")
            return

        # Load and preprocess the image (use cropped if available)
        img = self.cropped_image if self.cropped_image is not None else cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.features_label.config(text="Error: Could not read the image. Check the file path or integrity.")
            return

        if len(img.shape) == 3:  # Convert to grayscale if it's a color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image
        _, thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

        # Segmentation and feature extraction
        height, width = thresholded.shape
        segment_width = width // segments

        absolute_vector = []
        for i in range(segments):
            segment = thresholded[:, i * segment_width:(i + 1) * segment_width]
            count_black_pixels = np.sum(segment == 0)
            absolute_vector.append(count_black_pixels)

        # Normalization
        sum_normalized_vector = [x / sum(absolute_vector) for x in absolute_vector]
        max_normalized_vector = [x / max(absolute_vector) for x in absolute_vector]

        # Display feature vectors
        self.features_label.config(text=f"Absolute Vector: {[f'{x:.2f}' for x in absolute_vector]}\n" +
                                        f"Shkliar S1: {[f'{x:.2f}' for x in sum_normalized_vector]}\n" +
                                        f"Shkliar M1: {[f'{x:.2f}' for x in max_normalized_vector]}")

        # Display thresholded image with segment lines
        self.display_processed_image(thresholded, segments)

    def display_processed_image(self, img_array, segments):
        # Convert to color image for displaying segment lines
        img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Draw vertical lines for each segment
        height, width = img_color.shape[:2]
        segment_width = width // segments
        for i in range(1, segments):
            x = i * segment_width
            cv2.line(img_color, (x, 0), (x, height), (0, 0, 255), 2)

        # Convert to PIL format for Tkinter
        img = Image.fromarray(img_color)
        img = img.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_image)


root = tk.Tk()
app = FeatureExtractionApp(root)
root.mainloop()
