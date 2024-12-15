import cv2
import os


class TextDetection:
    def __init__(self, padding=6, output_dir="saved_rois"):
        """
        Initialize the TextDetection class with a specified padding size for bounding boxes.
        Args:
            padding (int): Padding size for bounding boxes.
            output_dir (str): Default directory where ROIs will be saved.
        """
        self.padding = padding
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_text(self, image_paths):
        """
        Detect text in one or multiple images and return bounding box information.
        
        Args:
            image_paths (str or list): Path to a single image file or a list of image paths.
        
        Returns:
            dict: A dictionary mapping image paths to their bounding boxes.
                Example: {"image1.jpg": [(x_min, y_min, x_max, y_max), ...], ...}
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]  # Convert single image path to a list


        from paddleocr import PaddleOCR 
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

        results_dict = {}

        for image_path in image_paths:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Invalid image file: {image_path}")

            # Run OCR detection on the image
            results = self.ocr.ocr(image_path, det=True, rec=False)

            bboxes = []
            if results[0]:
                for bbox in results[0]:
                    # Extract x and y coordinates from the bounding box points
                    x_coords = [int(point[0]) for point in bbox]
                    y_coords = [int(point[1]) for point in bbox]

                    # Calculate the bounding box limits with padding
                    x_min = max(0, min(x_coords) - self.padding)
                    x_max = min(image.shape[1], max(x_coords) + self.padding)
                    y_min = max(0, min(y_coords) - self.padding)
                    y_max = min(image.shape[0], max(y_coords) + self.padding)

                    bboxes.append((x_min, y_min, x_max, y_max))

            results_dict[image_path] = bboxes

        return results_dict


    def save_rois(self, image_path, bboxes, image_output_dir=None):
        """
        Save cropped ROIs (Regions of Interest) based on bounding boxes to the specified output directory.
        
        Args:
            image_path (str): Path to the image file.
            bboxes (list): List of bounding boxes [(x_min, y_min, x_max, y_max), ...].
            image_output_dir (str): Optional directory to save the ROIs for this specific image.
                                    If None, the default output directory is used.
        """
        # Use the specified subdirectory or default output directory
        if image_output_dir is None:
            image_output_dir = self.output_dir

        # Ensure the subdirectory exists
        os.makedirs(image_output_dir, exist_ok=True)

        # Sort bounding boxes by (y_min, x_min) for consistent ordering
        bboxes_sorted = sorted(bboxes, key=lambda box: (box[1], box[0]))

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Invalid image file: {image_path}")

        # Save each ROI as a separate image file in the specified subdirectory
        for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes_sorted):
            roi = image[y_min:y_max, x_min:x_max]
            roi_path = os.path.join(image_output_dir, f"roi_{i}.png")
            cv2.imwrite(roi_path, roi)

        print(f"Saved {len(bboxes_sorted)} ROIs to {image_output_dir}")




