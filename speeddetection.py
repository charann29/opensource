from ultralytics import YOLO
import cv2
from time import time
import numpy as np
from Pyresearch import BirdsEyeView
import colorsys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os


# getting Video
cap = cv2.VideoCapture("2103099-uhd_3840_2160_30fps.mp4") #0 and 1 webcam
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
source_fps = int(cap.get(cv2.CAP_PROP_FPS))
print("frame_width: ", frame_width)
print("frame_height: ", frame_height)
print("Source FPS: ", source_fps)
if not cap.isOpened():
    print("Error Opening Video File.")

# Saving Video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("output_video.mp4", fourcc, source_fps, (frame_width, frame_height))


# getting color for each track_id
def color(tracking_id):
    hue = (tracking_id * 137.5) % 360  # Change 137.5 to adjust the hue spread
    red, green, brown = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    return int(red * 255), int(green * 255), int(brown * 255)


# Loading Yolo V8 Model
model = YOLO("yolov8n.pt")

# Road polygons to be removed from tracking
road_polygon_points = [(0, 0), (0, 789), (3300, int(789)),
                       (3300, 0), (0, 0)]

# Perspective Transform from Camera view to Birds Eye View
target_width = 50
target_height = 250
source = np.array([[1252, 789], [2289, 789], [5039, 2159], [-550, 2159]])
target = np.array([[0, 0], [target_width-1, 0], [target_width-1, target_height-1], [0, target_height-1]])
transformation = BirdsEyeView(source, target, frame_width, frame_height, target_width, target_height)

# Declaration
prev_y_dict = {}  # Dictionary to store previous y-coordinate to calculate speeds
speeding_vehicles = {}  # Dictionary to store speeding vehicles' images and speeds
trails = {}  # Dictionary to store the trails for each track_id


# Create a directory to store speeding vehicle images
output_dir = "speeding_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ptime = 0  # Start Time to Calculate fps

while True:
    rect, frame = cap.read()

    if not rect:
        break
    frame_for_ticket = frame.copy()
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(road_polygon_points, dtype=np.int32)], (255, 255, 255))
    original_region = cv2.bitwise_and(frame, mask)
    mask_inv = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, mask_inv)
    frame = transformation.draw_road(frame)

    results = model.track(frame, conf=0.3, imgsz=(3840, 2176),
                          persist=True, classes=[2, 7], tracker="bytetrack.yaml", verbose=False)

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, track_id, conf, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            track_id = int(track_id)
            class_id = int(class_id)
            color_id = color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 5)
            cv2.putText(frame, ("car" if class_id == 2 else "truck" if class_id == 7 else None) + f" - {track_id}",
                        (x1, y1 - 10), 1, cv2.FONT_HERSHEY_COMPLEX, color_id, 3)

            # Drawing trail behind the vehicle
            if track_id not in trails:
                trails[track_id] = []
            trails[track_id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            if len(trails[track_id]) > 10:  # Limiting the length of the trail
                trails[track_id].pop(0)
            for i in range(1, len(trails[track_id])):
                cv2.line(frame, trails[track_id][i - 1], trails[track_id][i], color_id, 3)

            # Transformation and drawing on canvas
            bottom_center_point_bbox = (int((x1 + x2) / 2), y2)
            cv2.circle(frame, bottom_center_point_bbox, 10, color_id, -1)
            bottom_center_point_array = np.array([bottom_center_point_bbox])
            transformed_points = transformation.transform_points(points=bottom_center_point_array)[0]
            frame = transformation.draw_car_point(transformed_points, frame, track_id, color_id)

            # Speed Calculation
            if track_id not in prev_y_dict:
                prev_y_dict[track_id] = [transformed_points[1]]
            else:
                prev_y_dict[track_id].append(transformed_points[1])
                speed = transformation.speed_calculation(prev_y_dict[track_id], source_fps)
                if len(prev_y_dict[track_id]) > (source_fps / 2):
                    prev_y_dict[track_id].pop(0)

                # Labeling Speed Calculation on the car and canvas
                label_width = int((x2 - x1) * 0.45)
                bbox_center = int(((x1 + x2) / 2))
                cv2.rectangle(frame, (bbox_center-label_width, y2+10), (bbox_center+label_width, y2 + 40), color_id, -1)
                if label_width > 115:
                    text_scale = 0.5
                else:
                    text_scale = 0.95
                cv2.putText(frame, f"{speed} Km/h.",
                            (int(bbox_center-label_width * text_scale), y2+33), 4, cv2.FONT_HERSHEY_PLAIN, (0, 0, 0), 2)
                transformation.label_speed_on_canvas(speed, frame, color_id)

                # Ticket Generation for cars with speed more than 120 km/h and trucks more than 100 km/h
                if (class_id == 2 and speed > 120) or (class_id == 7 and speed > 100):
                    if track_id not in speeding_vehicles:
                        speeding_vehicles[track_id] = {"speed": speed, "images": []}

                        # Save the image of the speeding vehicle along with its speed
                    image_filename = f"speeding_vehicle_{track_id}_{speed}Kmph.jpg"
                    cv2.imwrite(os.path.join(output_dir, image_filename), frame_for_ticket[y1:y2, x1:x2])
                    speeding_vehicles[track_id]["images"].append((image_filename, speed))

    frame = cv2.add(frame, original_region)
    ctime = time()
    fps = round((1 / (ctime - ptime)), 2)
    ptime = ctime
    cv2.putText(frame, f"Pyresearch: FPS: {fps}", (30, 40), 4, cv2.FONT_HERSHEY_PLAIN, (0, 225, 0), 2)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# After video processing ends, generate PDFs for each speeding vehicle
for track_id, vehicle_data in speeding_vehicles.items():
    pdf_filename = f"speeding_vehicle_{track_id}_ticket.pdf"
    c = canvas.Canvas(os.path.join(output_dir, pdf_filename), pagesize=letter)

    # Add details to the PDF
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Vehicle ID: {track_id}")
    c.drawString(100, 730, f"Speed: {vehicle_data['speed']} Km/h")

    # Draw images of the speeding vehicle along with speed
    image_y = 500

    for image_filename, speed in vehicle_data["images"]:
        c.drawString(100, image_y - 20, f"Speed: {speed} Km/h")  # Add speed information
        c.drawImage(os.path.join(output_dir, image_filename), 100, image_y, width=400, height=250)
        image_y -= 300
        # Add the logo to the PDF
    
    logo_path = "Transparent logo.png"  # Replace with the actual path to your logo image
    c.drawImage(logo_path, 100, 50, width=100, height=50)  # Adjust the coordinates and size as needed
    

    # Save and close the PDF
    c.save()