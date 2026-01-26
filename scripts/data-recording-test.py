# Generated using Gemini
# Remove this file after first implementation

import cosysairsim as airsim
import cv2
import time
import os
import csv
import numpy as np

# --- SETTINGS ---
STORAGE_DIR = "c_data"
frame_rate = 10  # How many images per second to save

# Create the folder if it doesn't exist
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
    print(f"Created folder: {STORAGE_DIR}")

# Create/Open the CSV file to log steering data
csv_file_path = os.path.join(STORAGE_DIR, "log.csv")
# We open in 'a' (append) mode so we don't delete old data if we restart
csv_file = open(csv_file_path, 'a', newline='')
writer = csv.writer(csv_file)

# Connect to the car
client = airsim.CarClient()
client.confirmConnection()

# We do NOT enable API control because YOU are driving manually
# client.enableApiControl(False) 

print("Connected! Start driving manually.")
print(f"Recording data to '{STORAGE_DIR}'... Press Ctrl+C to stop.")

try:
    image_count = 0
    while True:
        # 1. Get the current state of the car (Steering, Throttle, Speed)
        car_state = client.getCarState()
        steering = car_state.kinematics_estimated.orientation # This is raw quaternion, simpler to use controls
        
        # Better way: Get the actual controls you are pressing
        controls = client.getCarControls()
        steering_angle = controls.steering
        throttle = controls.throttle
        brake = controls.brake
        speed = car_state.speed

        # 2. Get the image
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]

        # 3. Process the image
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # 4. Save the file
        # Use a unique filename based on time to avoid overwrites
        timestamp = int(time.time() * 1000)
        filename = f"img_{timestamp}.png"
        filepath = os.path.join(STORAGE_DIR, filename)
        
        cv2.imwrite(filepath, img_rgb)

        # 5. Write the data to CSV
        # Format: [ImageName, Steering, Throttle, Brake, Speed]
        writer.writerow([filename, steering_angle, throttle, brake, speed])
        
        image_count += 1
        if image_count % 10 == 0:
            print(f"Recorded {image_count} frames...")

        # Sleep to match frame rate (0.1s = 10fps)
        time.sleep(1.0 / frame_rate)

except KeyboardInterrupt:
    print("\nStopped recording.")
    csv_file.close()
    print(f"Data saved to {STORAGE_DIR}/log.csv")