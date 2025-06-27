# test_oakd_camera.py
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

try:
    response = client.simGetImages([
        airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False)
    ])
    print("Success! Got image of size:", response[0].width, "x", response[0].height)
except Exception as e:
    print("Camera access failed:", e)
