# Configuration constants for UAV navigation

MAX_FLOW_MAG = 50.0
MAX_VECTOR_COMPONENT = 20.0
MIN_FEATURES_PER_ZONE = 10
GRACE_PERIOD_SEC = 1.0
MAX_SIM_DURATION = 60  # seconds
GOAL_X = 29  # meters
GOAL_RADIUS = 1.0
MIN_PROBE_FEATURES = 5
TARGET_FPS = 20
LOG_INTERVAL = 5
VIDEO_FPS = 8.0
VIDEO_SIZE = (1280, 720)
VIDEO_OUTPUT = 'flow_output.avi'

def load_app_config(config_path: str = "config.ini"):
    """Load application config from an INI file."""
    import configparser
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser
