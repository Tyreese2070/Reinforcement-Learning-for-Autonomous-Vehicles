# Activate venv for windows terminal: .\\venv\\Scripts\\activate
# Generated using Gemini
# Remove this file after first implementation
import cosysairsim as airsim
import time

# 1. Connect to the simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True) # Take control from the human/keyboard

print("Connected! Car is stopping...")
car_controls = airsim.CarControls()
car_controls.brake = 1.0
car_controls.throttle = 0.0
client.setCarControls(car_controls)

time.sleep(2)

print("Go go go!")
car_controls.brake = 0.0
car_controls.throttle = 1.0
client.setCarControls(car_controls)