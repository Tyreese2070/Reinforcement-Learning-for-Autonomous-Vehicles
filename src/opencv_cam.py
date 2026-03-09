import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

def test_camera_simple():

    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        # Define the camera and add it to the screen
        sensors=dict(rgb_camera=(RGBCamera, 256, 256)),
        interface_panel=["rgb_camera"],
        image_observation=True, 
    )

    env = MetaDriveEnv(config)
    env.reset()
    
    print("Drive with W/A/S/D. The OpenCV window should appear now.")

    for i in range(5000):
        obs, reward, terminated, truncated, info = env.step([0, 0])
        
        # Get image from observation
        if isinstance(obs, dict):
             image_data = obs.get("image", obs.get("rgb_camera"))
        else:
             image_data = obs

        # Process image
        if image_data is not None:
            # Scale 0-1 float to 0-255 uint8
            img_uint8 = (image_data * 255).astype(np.uint8)
            
            img_uint8 = img_uint8.squeeze()
            
            if len(img_uint8.shape) == 3:
                img_display = cv2.resize(img_uint8, (512, 512)) # resize
                
                # Convert RGB to BGR for opencv
                img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                
                cv2.imshow("Display", img_display)
                cv2.waitKey(1)

        env.render()
        
        if terminated or truncated:
            env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_simple()