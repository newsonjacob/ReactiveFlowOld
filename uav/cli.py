import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Optical flow navigation script")
    parser.add_argument("--manual-nudge", action="store_true", help="Enable manual nudge at frame 5 for testing")
    parser.add_argument("--map", choices=["reactive", "deliberative", "hybrid"], default="reactive", help="Which map to load")
    parser.add_argument("--ue4-path", default=None, help="Override the default path to the Unreal Engine executable")
    parser.add_argument("--settings-path", default=None, help="Path to AirSim settings.json")
    parser.add_argument("--config", default="config.ini", help="Path to config file with default paths")
    parser.add_argument("--goal-x", type=int, default=29, help="Distance from start to goal (X coordinate)")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum simulation duration in seconds")
    return parser.parse_args()
