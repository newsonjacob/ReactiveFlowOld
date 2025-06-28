from uav.cli import parse_args
from uav.sim_launcher import launch_sim
from uav.nav_loop import run_navigation
import airsim

SETTINGS_PATH = r"C:\Users\Jacob\Documents\AirSim\settings.json"

def main():
    args = parse_args()
    sim_process = launch_sim(args, SETTINGS_PATH)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    run_navigation(args, client, sim_process, SETTINGS_PATH)

if __name__ == "__main__":
    main()