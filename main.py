import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from uav.cli import parse_args
from uav.sim_launcher import launch_sim
from uav.nav_loop import (setup_environment, start_perception_thread, navigation_loop, cleanup)
import airsim
from uav.utils import FLOW_STD_MAX
from uav.config import load_app_config

from pathlib import Path

# Default path to AirSim settings file
SETTINGS_PATH = str(Path.home() / "Documents" / "AirSim" / "settings.json")

def get_settings_path(args, config):
    """
    Determine the path to the AirSim settings file.
    Priority: command-line argument > config file > default path.
    """
    try:
        return args.settings_path or config.get("paths", "settings")
    except Exception:
        return SETTINGS_PATH

def main() -> None:
    # Parse command-line arguments (e.g., config file, simulation settings)
    args = parse_args()
    # Load application configuration from file
    config = load_app_config(args.config)
    # Get the AirSim settings path
    settings_path = get_settings_path(args, config)
    # Launch the AirSim simulator process
    sim_process = launch_sim(args, settings_path, config)

    # Wait for the simulator to be ready before connecting the AirSim client
    import time
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            client = airsim.MultirotorClient()  # Create AirSim client
            client.confirmConnection()          # Try to connect
            break                              # Success: exit loop
        except Exception as e:
            logging.info(f"Waiting for simulator to be ready... (attempt {attempt+1}/{max_attempts})")
            time.sleep(2)                      # Wait and retry
    else:
        # If connection fails after all attempts, clean up and exit
        logging.error("Failed to connect to AirSim simulator after multiple attempts.")
        cleanup(None, sim_process, None)
        return

    # Enable API control and arm the drone (redundant for safety)
    client.enableApiControl(True)
    client.armDisarm(True)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    ctx = None  # Context dictionary for sharing state between modules
    try:
        # Set up the simulation environment (logging, video, navigation state, etc.)
        ctx = setup_environment(args, client)
        # Start the perception thread (handles image capture and optical flow)
        start_perception_thread(ctx)
        # Enter the main navigation loop (handles drone movement and logic)
        navigation_loop(args, client, ctx)
    finally:
        # Always clean up resources (stop threads, close files, terminate sim)
        cleanup(client, sim_process, ctx if ctx is not None else None)

    # Note: navigator.settling is handled inside the navigation loop

if __name__ == "__main__":
    main()