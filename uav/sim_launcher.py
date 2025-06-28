import subprocess
import time
import logging
from configparser import ConfigParser

logger = logging.getLogger(__name__)

def launch_sim(args, settings_path, config: ConfigParser | None = None):
    map_launch_args = {
        "reactive": "/Game/Maps/Map_Reactive",
        "deliberative": "/Game/Maps/Map_Deliberative",
        "hybrid": "/Game/Maps/Map_Hybrid"
    }

    exe_paths = {
        "reactive": r"H:\\Documents\\AirSimBuilds\\Reactive\\WindowsNoEditor\\Blocks\\Binaries\\Win64\\Blocks.exe",
        "deliberative": r"H:\\Documents\\AirSimBuilds\\Deliberative\\WindowsNoEditor\\Blocks\\Binaries\\Win64\\Blocks.exe",
        "hybrid": r"H:\\Documents\\AirSimBuilds\\Hybrid\\WindowsNoEditor\\Blocks\\Binaries\\Win64\\Blocks.exe"
    }

    config_exe = None
    if config is not None:
        try:
            config_exe = config.get("ue4", args.map)
        except Exception:
            config_exe = None

    ue4_exe = args.ue4_path or config_exe or exe_paths[args.map]
    map_path = map_launch_args[args.map]

    sim_cmd = [
        ue4_exe,
        "-windowed",
        "-ResX=1280",
        "-ResY=720",
        f'-settings={settings_path}',
        map_path
    ]

    logger.info("\U0001F680 Launching UE4 map '%s'...", args.map)
    sim_process = subprocess.Popen(sim_cmd)
    time.sleep(5)  # Give UE4 time to boot up
    return sim_process
