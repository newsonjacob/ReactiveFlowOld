from uav.cli import parse_args
from uav.sim_launcher import launch_sim
from uav.nav_loop import (
    setup_environment,
    start_perception_thread,
    navigation_loop,
    cleanup,
)
import airsim
import logging
from uav.utils import FLOW_STD_MAX as UTIL_FLOW_STD_MAX
from uav.config import load_app_config

FLOW_STD_MAX = UTIL_FLOW_STD_MAX

SETTINGS_PATH = r"C:\Users\Jacob\Documents\AirSim\settings.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_app_config(args.config)
    settings_path = args.settings_path or config.get("paths", "settings", fallback=SETTINGS_PATH)
    sim_process = launch_sim(args, settings_path, config)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    ctx = setup_environment(args, client)
    start_perception_thread(ctx)
    try:
        navigation_loop(args, client, ctx)
    finally:
        cleanup(client, sim_process, ctx)

    # navigator.settling handled in nav_loop


if __name__ == "__main__":
    main()

