import cv2
import pytesseract
import numpy as np
import csv
from concurrent.futures import ThreadPoolExecutor

# Set up Tesseract executable path if not added to system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path based on your Tesseract installation

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to enhance text detection (optional, can be adjusted based on your needs)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply Gaussian blur to reduce noise and improve OCR accuracy
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def process_frame(frame, frame_number, frame_rate):
    binary = preprocess_frame(frame)
    # Apply OCR to the preprocessed frame
    text = pytesseract.image_to_string(binary, config='--psm 6')  # Use Page Segmentation Mode 6 (single uniform block of text)
    # If text is found, record the timestamp and text
    if text.strip():  # Check if text is not empty
        timestamp = frame_number / frame_rate
        return timestamp, text.strip()
    return None

def process_video(video_path, output_csv, frame_skip=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    
    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp (s)', 'Detected Text'])
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_number % frame_skip == 0:
                    futures.append(executor.submit(process_frame, frame, frame_number, frame_rate))
                
                print(f"Processing frame {frame_number}")
                frame_number += 1
            
            for future in futures:
                result = future.result()
                if result:
                    writer.writerow(result)
                    print(f"Text found at {result[0]:.2f} seconds: {result[1]}")
    
    cap.release()

# Example usage
video_path = './vids/calendar.mp4'
output_csv = './output/cal2.csv'
process_video(video_path, output_csv, frame_skip=120)
print(f"Text detection completed. Timestamps and text saved in {output_csv}")
