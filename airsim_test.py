import airsim

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
