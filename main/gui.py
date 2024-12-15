import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from multiprocessing import Process
import json
import os
from deskew_img import straighten_img


# Define text detection process
def text_detection_process(image_path, output_dir, result_file):
    """Isolated text detection process."""
    from text_detection import TextDetection  # Import inside subprocess
    detector = TextDetection(padding=6, output_dir=output_dir)
    bboxes = detector.detect_text(image_path)
    with open(result_file, "w") as f:
        json.dump(bboxes, f)


def text_recognition_process(roi_dir, output_file, image_path, bboxes, create_word=True, create_pdf=True):
    """
    Isolated text recognition process with options for Word and PDF export.
    
    Args:
        roi_dir (str): Directory containing Regions of Interest (ROIs).
        output_file (str): Path for the Word output file.
        image_path (str): Path to the original image.
        bboxes (list): List of bounding boxes [(x_min, y_min, x_max, y_max), ...].
        create_word (bool): Whether to create the Word document. Default is True.
        create_pdf (bool): Whether to create the PDF. Default is True.
    """
    from text_recognition import TrOCRProcessorTool  # Import inside subprocess
    recognizer = TrOCRProcessorTool(roi_dir=roi_dir, output_word_file=output_file)

    # Perform OCR on ROIs, saving the Word file only if requested
    ocr_results = recognizer.process_rois(save_word=create_word)

    # Conditionally create the PDF
    if create_pdf:
        recognizer.save_to_pdf_with_layout(image_path, bboxes, ocr_results, output_pdf="ocr_results.pdf")

def text_detection_batch_process(folder_path, output_dir, result_file):
    """
    Isolated text detection subprocess for a folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
        output_dir (str): Directory to save ROIs.
        result_file (str): JSON file to store bounding boxes for all images.
    """
    from text_detection import TextDetection  # Import inside subprocess

    detector = TextDetection(padding=6, output_dir=output_dir)

    # Collect all image paths
    image_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    print(f"Found {len(image_paths)} images to process.")

    all_bboxes = detector.detect_text(image_paths)

    for image_path, bboxes in all_bboxes.items():
        # Create a subdirectory for each image
        image_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(image_output_dir, exist_ok=True)

        if bboxes:
            print(f"Saving {len(bboxes)} ROIs for {image_path} in {image_output_dir}.")
            detector.save_rois(image_path, bboxes, image_output_dir)
        else:
            print(f"No ROIs detected for {image_path}.")

    # Save all bounding boxes to a JSON file
    with open(result_file, "w") as f:
        json.dump(all_bboxes, f)

    print(f"Text detection completed. Results saved to {result_file}.")


def text_recognition_batch_process(roi_base_dir, result_file, output_csv):
    """
    Isolated text recognition subprocess for a folder of ROIs.
    
    Args:
        roi_base_dir (str): Base directory containing ROIs for each image.
        result_file (str): JSON file containing bounding boxes for all images.
        output_csv (str): Path to the output CSV file.
    """
    from text_recognition import TrOCRProcessorTool  # Import inside subprocess
    import pandas as pd

    # Load bounding boxes from the detection results
    with open(result_file, "r") as f:
        all_bboxes = json.load(f)

    all_results = []  # List to store OCR results for CSV output

    # Initialize a single TrOCRProcessorTool instance
    recognizer = TrOCRProcessorTool(roi_dir=None, output_word_file=None)

    # Process each folder of ROIs
    for image_name, bboxes in all_bboxes.items():
        roi_dir = os.path.join(roi_base_dir, os.path.splitext(os.path.basename(image_name))[0])
        print(f"Processing ROIs for {image_name} in {roi_dir}")

        if not os.path.exists(roi_dir):
            print(f"Skipping {image_name}: ROI directory does not exist.")
            continue

        # Temporarily update the ROI directory in the recognizer
        recognizer.roi_dir = roi_dir

        # Process ROIs for the current image
        ocr_results = recognizer.process_rois(save_word=False)

        # Debug: Print the OCR results for this image
        print(f"OCR results for {image_name}: {ocr_results}")

        # Collect results for CSV
        for roi_name, text in ocr_results.items():
            all_results.append({"Image": image_name, "ROI": roi_name, "Text": text})

    # Debug: Print the collected OCR results
    print("Collected OCR results:", all_results)

    # Save results to a CSV file
    if all_results:  # Only save if there are results
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"OCR results saved to {output_csv}")
    else:
        print(f"No OCR results found. The CSV file will not be created.")

class ImageDrawingTool:
    def __init__(self, root):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = root
        self.root.geometry("1000x600")  # Set a fixed size for the GUI
        self.root.title("Image Drawing Tool")

        # Initialize variables
        self.file_path = ""
        self.original_image = None
        self.resized_image = None
        self.resized_width = 0
        self.resized_height = 0
        self.bboxes = []
        self.selected_rois = []
        self.recognize_button = None

        self.text_detector_output_dir = "saved_rois"
        self.text_recognition_output_file = "ocr_results.docx"
        self.temp_result_file = "temp_bboxes.json"

        self.create_widgets()

    def create_widgets(self):
        self.canvas_frame = ctk.CTkFrame(self.root)
        self.canvas_frame.pack(side="top", expand=True, fill="both")

        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="#1E1E1E")
        self.canvas.pack(side="left", expand=True, fill="both")

        self.v_scroll = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.v_scroll.pack(side="right", fill="y")

        self.h_scroll = ctk.CTkScrollbar(self.canvas_frame, orientation="horizontal", command=self.canvas.xview)
        self.h_scroll.pack(side="bottom", fill="x")

        self.canvas.config(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.button_frame = ctk.CTkFrame(self.root)
        self.button_frame.pack(side="bottom", pady=10)

        self.image_button = ctk.CTkButton(self.button_frame, text="Add image", command=self.add_image, width=120)
        self.batch_button = ctk.CTkButton(self.button_frame, text="OCR for folder", command=self.batch_process, width=160)
        self.detect_button = ctk.CTkButton(self.button_frame, text="Detect Text", command=self.detect_text, width=120)
        self.save_button = ctk.CTkButton(self.button_frame, text="Save selected ROIs", command=self.save_selected_rois, width=160)

        self.image_button.pack(side="left", padx=10)
        self.batch_button.pack(side="left", padx=10)
        self.detect_button.pack(side="left", padx=10)
        self.save_button.pack(side="left", padx=10)

        # Placeholder buttons that will appear only after ROIs are saved
        self.recognize_button = None
        self.export_with_layout_button = None

    def add_image(self):
        self.file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff"), ("All files", "*.*")]
        )
        if not self.file_path:
            return

        # Hide the recognize and export buttons when a new image is added
        if self.recognize_button:
            self.recognize_button.pack_forget()  # Remove the button from the layout
            self.recognize_button = None  # Reset the button variable

        if self.export_with_layout_button:
            self.export_with_layout_button.pack_forget()  # Remove the button from the layout
            self.export_with_layout_button = None  # Reset the button variable

        self.file_path = straighten_img(self.file_path)
        self.original_image = Image.open(self.file_path)
        self.update_image_on_canvas()


    def update_image_on_canvas(self):
        if self.original_image:
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            image_width, image_height = self.original_image.size
            aspect_ratio = image_width / image_height

            if canvas_width / canvas_height > aspect_ratio:
                new_width = int(canvas_height * aspect_ratio)
                new_height = canvas_height
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)

            self.resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.resized_width, self.resized_height = self.resized_image.size

            self.canvas.config(scrollregion=(0, 0, self.resized_width, self.resized_height))
            self.image_tk = ImageTk.PhotoImage(self.resized_image)

            # Center the image on the canvas
            canvas_center_x = self.canvas.winfo_width() / 2
            canvas_center_y = self.canvas.winfo_height() / 2
            self.canvas.delete("all")
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.image_tk, anchor="center")

    def detect_text(self):
        if not self.file_path:
            self.show_custom_messagebox("Error", "Please load an image first!")
            return

        detection_process = Process(
            target=text_detection_process,
            args=(self.file_path, self.text_detector_output_dir, self.temp_result_file)
        )
        detection_process.start()
        detection_process.join()

        if os.path.exists(self.temp_result_file):
            with open(self.temp_result_file, "r") as f:
                result_dict = json.load(f)
                self.bboxes = result_dict.get(self.file_path, [])  # Extract bounding boxes for the current image
            os.remove(self.temp_result_file)

        self.canvas.delete("all")
        if self.resized_image:
            canvas_center_x = self.canvas.winfo_width() / 2
            canvas_center_y = self.canvas.winfo_height() / 2
            self.image_tk = ImageTk.PhotoImage(self.resized_image)
            image_id = self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.image_tk, anchor="center")

            image_offset_x = canvas_center_x - (self.resized_width / 2)
            image_offset_y = canvas_center_y - (self.resized_height / 2)

            # Draw bounding boxes
            for i, (x_min, y_min, x_max, y_max) in enumerate(self.bboxes):
                x_min_resized = x_min / self.original_image.width * self.resized_width + image_offset_x
                y_min_resized = y_min / self.original_image.height * self.resized_height + image_offset_y
                x_max_resized = x_max / self.original_image.width * self.resized_width + image_offset_x
                y_max_resized = y_max / self.original_image.height * self.resized_height + image_offset_y

                self.canvas.create_rectangle(
                    x_min_resized, y_min_resized, x_max_resized, y_max_resized,
                    outline="red", width=2, tags=f"roi_{i}"
                )

        self.canvas.bind("<Button-1>", self.handle_click)



    def handle_click(self, event):
        # Get the click position on the canvas
        click_x = self.canvas.canvasx(event.x)
        click_y = self.canvas.canvasy(event.y)

        # Calculate the image's offset on the canvas
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        image_offset_x = canvas_center_x - (self.resized_width / 2)
        image_offset_y = canvas_center_y - (self.resized_height / 2)

        # Check if the click falls within any of the bounding boxes
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.bboxes):
            # Adjust bounding box coordinates to account for image offset
            x_min_resized = x_min / self.original_image.width * self.resized_width + image_offset_x
            y_min_resized = y_min / self.original_image.height * self.resized_height + image_offset_y
            x_max_resized = x_max / self.original_image.width * self.resized_width + image_offset_x
            y_max_resized = y_max / self.original_image.height * self.resized_height + image_offset_y

            # Check if the click is inside the bounding box
            if x_min_resized <= click_x <= x_max_resized and y_min_resized <= click_y <= y_max_resized:
                self.select_roi(i)
                break


    def select_roi(self, index):
        if index in self.selected_rois:
            self.selected_rois.remove(index)
            self.canvas.itemconfig(f"roi_{index}", outline="red")
        else:
            self.selected_rois.append(index)
            self.canvas.itemconfig(f"roi_{index}", outline="blue")

    def save_selected_rois(self):
        if not self.selected_rois:
            self.show_custom_messagebox("Error", "No ROIs selected!")
            return

        # Clear the output directory before saving
        if os.path.exists(self.text_detector_output_dir):
            for file in os.listdir(self.text_detector_output_dir):
                file_path = os.path.join(self.text_detector_output_dir, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        else:
            os.makedirs(self.text_detector_output_dir, exist_ok=True)

        # Sort the selected bounding boxes (top-to-bottom, left-to-right)
        selected_bboxes = [self.bboxes[i] for i in self.selected_rois]
        sorted_bboxes = sorted(selected_bboxes, key=lambda box: (box[1], box[0]))  # Sort by y_min, then x_min

        # Import TextDetection locally to avoid torch conflict
        from text_detection import TextDetection
        detector = TextDetection(padding=6, output_dir=self.text_detector_output_dir)
        detector.save_rois(self.file_path, sorted_bboxes)

        # Save sorted bounding boxes to a JSON file
        coordinates_file = os.path.join("saved_coordinates.json")
        with open(coordinates_file, "w") as f:
            json.dump(sorted_bboxes, f)

        self.bboxes = sorted_bboxes  # Update self.bboxes with the sorted bounding boxes

        self.show_custom_messagebox("Success", f"Selected ROIs saved to {self.text_detector_output_dir}")

        # Make "Recognize Text" and "Export with Layout" buttons visible
        if not self.recognize_button:
            self.recognize_button = ctk.CTkButton(
                self.button_frame, text="Export to Word", command=self.recognize_text, width=140
            )
            self.recognize_button.pack(side="left", padx=10)

        if not self.export_with_layout_button:
            self.export_with_layout_button = ctk.CTkButton(
                self.button_frame, text="Export to PDF", command=self.export_with_layout, width=180
            )
            self.export_with_layout_button.pack(side="right", padx=10)


    def recognize_text(self):
        if not os.listdir(self.text_detector_output_dir):
            self.show_custom_messagebox("Error", "No ROIs found to process!")
            return

        recognition_process = Process(
            target=text_recognition_process,
            args=(self.text_detector_output_dir, self.text_recognition_output_file, self.file_path, self.bboxes, True, False)
        )
        recognition_process.start()
        recognition_process.join()

        self.show_custom_messagebox("Success", "Text recognition with layout completed! Results saved to ocr_results.docx")

    def export_with_layout(self):
        if not os.listdir(self.text_detector_output_dir):
            self.show_custom_messagebox("Error", "No ROIs found to process!")
            return

        recognition_process = Process(
            target=text_recognition_process,
            args=(self.text_detector_output_dir, self.text_recognition_output_file, self.file_path, self.bboxes, False, True)
        )
        recognition_process.start()
        recognition_process.join()

        self.show_custom_messagebox("Success", "Export with layout completed! Results saved to ocr_results.pdf")

    def batch_process(self):
        folder_path = filedialog.askdirectory(title="Select Folder of Images")
        if not folder_path:
            self.show_custom_messagebox("Error", "No folder selected!")
            return

        roi_output_dir = os.path.join(folder_path, "batch_saved_rois")
        detection_result_file = os.path.join(folder_path, "detection_results.json")
        output_csv = os.path.join(folder_path, "ocr_results.csv")

        # Clear out the previous results
        if os.path.exists(roi_output_dir):
            import shutil
            shutil.rmtree(roi_output_dir)  # Remove the entire directory
        os.makedirs(roi_output_dir, exist_ok=True)

        if os.path.exists(detection_result_file):
            os.remove(detection_result_file)

        if os.path.exists(output_csv):
            os.remove(output_csv)

        # Subprocess for text detection
        detection_process = Process(
            target=text_detection_batch_process,
            args=(folder_path, roi_output_dir, detection_result_file)
        )
        detection_process.start()
        detection_process.join()

        # Subprocess for text recognition
        recognition_process = Process(
            target=text_recognition_batch_process,
            args=(roi_output_dir, detection_result_file, output_csv)
        )
        recognition_process.start()
        recognition_process.join()

        self.show_custom_messagebox("Success", f"Batch processing completed! Results saved to {output_csv}")



    def show_custom_messagebox(self, title, message):
        message_box = ctk.CTkToplevel(self.root)
        message_box.title(title)
        message_box.geometry("350x200")
        message_box.resizable(False, False)
        message_box.attributes("-topmost", True)  # Ensure the message box appears on top

        # Center the message box on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        message_box_x = (screen_width - 350) // 2
        message_box_y = (screen_height - 200) // 2
        message_box.geometry(f"350x200+{message_box_x}+{message_box_y}")

        label = ctk.CTkLabel(message_box, text=message, wraplength=300)
        label.pack(pady=20, padx=20)

        ok_button = ctk.CTkButton(message_box, text="OK", command=message_box.destroy)
        ok_button.pack(pady=10)



if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageDrawingTool(root)
    root.mainloop()
