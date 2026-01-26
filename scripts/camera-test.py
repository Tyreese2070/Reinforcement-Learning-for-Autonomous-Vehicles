# Generated using Gemini
# Remove this file after first implementation
import cosysairsim as airsim
import cv2 # OpenAI's image library
import time
import numpy as np

# Connect
client = airsim.CarClient()
client.confirmConnection()

print("Taking a photo in 3 seconds...")
time.sleep(3)

# Request a photo from the front center camera
# '0' is usually the front camera ID
# ImageType.Scene is the standard RGB visualization
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

response = responses[0]

# Convert the raw data to a format OpenCV understands
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(response.height, response.width, 3)

# Save it to check if it worked
cv2.imwrite("car_view.png", img_rgb)
print("Saved car_view.png! Go check your folder.")