import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from ultralytics import YOLO
from vehicle_classifier_model import VehicleClassifier

# Paths
video_path = "results/2103099-uhd_3840_2160_30fps.mp4"
output_path = "results/yolo_processed_video.mp4"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Vehicle classes
vehicle_classes = ["bicycle", "bus", "car", "motorbike"]

# YOLO class mapping
yolo_class_filter = {
    1: "bicycle",    # bicycle 
    2: "car",        # car 
    3: "motorbike",  # motorcycle 
    5: "bus",        # bus 
    7: "truck",      # truck 
}

# Classes to skip reclassification
skip_reclassify = [7]  # truck

# Colors for visualization
colors = {
    "bicycle": (0, 255, 0),     # Green
    "bus": (255, 0, 0),         # Blue
    "car": (0, 0, 255),         # Red
    "motorbike": (255, 255, 0), # Cyan
    "truck": (255, 165, 0),     # Orange
    "unknown": (128, 128, 128)  # Gray
}

# Load YOLOv8 model
def load_yolo_model():
    if not os.path.exists('yolov8n.pt'):
        download_yolo_model()
    model = YOLO('yolov8n.pt')
    return model

# Download YOLOv8 model if not exists
def download_yolo_model():
    print("Downloading YOLOv8 model...")
    try:
        import requests
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
        response = requests.get(url, stream=True)
        with open('yolov8n.pt', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Error downloading YOLOv8 model: {e}")
        raise

# Load vehicle classifier model
def load_classifier_model():
    try:
        model = VehicleClassifier(num_classes=len(vehicle_classes))
        checkpoint = torch.load('vehicle_classifier.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading vehicle classifier: {e}")
        return None

# Prediction transform
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classify vehicle using classifier
def classify_vehicle(frame, box, classifier_model):
    if classifier_model is None:
        return None, 0.0
        
    try:
        x1, y1, x2, y2 = box
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return None, 0.0
        
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = predict_transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, pred_class = torch.max(probabilities, 1)
            confidence = max_prob.item() * 100
            class_name = vehicle_classes[pred_class.item()]
            
        return class_name, confidence
    except Exception as e:
        return None, 0.0

# Process video
def process_video(yolo_model, classifier_model, video_path, output_path, confidence_threshold=0.4, max_frames=None, stride=2):
    # Initialize video capture and writer
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize for faster processing
    target_width = 1280
    target_height = int(height * (target_width / width))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/stride, (target_width, target_height))
    
    # Tracking metrics
    frame_count = 0
    processed_count = 0
    processing_times = []
    class_counts = {class_name: 0 for class_name in yolo_class_filter.values()}
    total_objects_detected = 0
    reclassifications = 0
    
    # Progress tracking
    start_time = time.time()
    report_interval = max(10, min(100, total_frames // 100)) if total_frames > 0 else 10
    
    # Process frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Skip frames according to stride
        if frame_count % stride != 0:
            frame_count += 1
            continue
            
        frame_start_time = time.time()
        frame = cv2.resize(frame, (target_width, target_height))
        
        # Run YOLO detection
        try:
            results = yolo_model(frame, conf=confidence_threshold, verbose=False)
            objects_in_frame = 0
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Process only vehicle classes
                    if cls_id in yolo_class_filter:
                        yolo_class = yolo_class_filter[cls_id]
                        objects_in_frame += 1
                        total_objects_detected += 1
                        
                        # Skip reclassification for truck
                        if cls_id in skip_reclassify:
                            class_name = yolo_class
                            confidence = conf * 100
                        else:
                            # Use classifier for other vehicles
                            refined_class, refined_conf = classify_vehicle(frame, (x1, y1, x2, y2), classifier_model)
                            
                            # Use threshold for classification
                            cls_threshold = 65 if yolo_class in ["bus", "motorbike"] else 75
                            
                            if refined_class and refined_conf > cls_threshold:
                                class_name = refined_class
                                confidence = refined_conf
                                if refined_class != yolo_class:
                                    reclassifications += 1
                            else:
                                class_name = yolo_class
                                confidence = conf * 100
                        
                        # Update class counts
                        class_counts[class_name] += 1
                        
                        # Draw bounding box and label
                        color = colors.get(class_name, colors["unknown"])
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.1f}%"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)-text_size[1]-10), 
                                    (int(x1)+text_size[0], int(y1)), 
                                    color, -1)
                        cv2.putText(frame, label, 
                                  (int(x1), int(y1)-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add class counts
            y_offset = 70
            for class_name, count in sorted(class_counts.items()):
                if count > 0:
                    color = colors.get(class_name, colors["unknown"])
                    cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset += 30
            
            # Write frame
            out.write(frame)
            
            # Track timing
            elapsed_time = time.time() - frame_start_time
            processing_times.append(elapsed_time)
            processed_count += 1
            
            # Display progress
            if frame_count % report_interval == 0:
                avg_time = sum(processing_times[-min(len(processing_times), 10):]) / min(len(processing_times), 10)
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                eta = (total_frames - frame_count) * avg_time / stride if total_frames > 0 else 0
                
                print(f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | "
                     f"Time: {avg_time:.3f}s/frame | ETA: {eta/60:.1f} min | "
                     f"Objects: {objects_in_frame}")
            
            # Check max_frames limit
            if max_frames and processed_count >= max_frames:
                break
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            
        frame_count += 1
    
    # Cleanup
    video.release()
    out.release()
    
    # Summary
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        total_time = time.time() - start_time
        
        print(f"\nVideo processing complete: {output_path}")
        print(f"Processed {processed_count}/{total_frames} frames")
        print(f"Average time: {avg_time:.3f}s/frame, Total time: {total_time/60:.2f} min")
        print(f"Total objects: {total_objects_detected}, Reclassifications: {reclassifications}")
        
        # Print class distribution
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / max(1, total_objects_detected) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in [colors.get(c, colors["unknown"]) for c in class_counts.keys()]]
        plt.bar(class_counts.keys(), class_counts.values(), color=colors_rgb)
        plt.title('Detected Vehicle Classes')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        percentages = [count / max(1, total_objects_detected) * 100 for count in class_counts.values()]
        plt.pie(percentages, labels=class_counts.keys(), autopct='%1.1f%%', colors=colors_rgb)
        plt.title('Vehicle Class Distribution')
        
        plt.tight_layout()
        plt.savefig('detected_vehicles.png')
        
        # Save basic stats to CSV
        try:
            import csv
            with open('detection_statistics.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Class', 'Count', 'Percentage (%)'])
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / max(1, total_objects_detected) * 100
                    writer.writerow([class_name, count, f"{percentage:.1f}"])
        except Exception as e:
            print(f"Error saving statistics: {e}")

def main():
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load models
    print(f"Using device: {device}")
    yolo_model = load_yolo_model()
    classifier_model = load_classifier_model()
    
    # Check video file
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Process video
    print(f"Processing video: {video_path}")
    full_output_path = "results/full_yolo_processed_video.mp4"
    
    process_video(
        yolo_model, 
        classifier_model, 
        video_path, 
        full_output_path, 
        confidence_threshold=0.4,
        max_frames=None,  # Process all frames
        stride=2           # Process every 2nd frame
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 