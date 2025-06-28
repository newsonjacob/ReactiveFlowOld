# Reactive Optical Flow UAV Navigation
[![CI](https://github.com/newsonjacob/ReactiveOptical_Flow/actions/workflows/tests.yml/badge.svg)](https://github.com/newsonjacob/ReactiveOptical_Flow/actions/workflows/tests.yml)

This repository contains the implementation of a reactive obstacle avoidance system for a simulated UAV using optical flow. The UAV operates in a GPS-denied environment simulated via AirSim and Unreal Engine 4.

---

## 📦 Features

- Real-time optical flow tracking using Lucas-Kanade method
- Zone-based obstacle avoidance using vector summation
- Logs flight data to CSV for post-flight analysis
- Visualises 3D flight paths with interactive HTML output
- Compatible with a custom stereo camera setup (`oakd_camera`)
- Works with auto-launched AirSim simulation

---

## Installation

```bash
pip install -e .
```

## 🚁 How It Works

- A drone is spawned in a UE4 + AirSim environment
- The system captures images from a front-facing virtual camera
- Optical flow vectors are extracted and used to detect motion in different zones
- The UAV avoids obstacles by adjusting yaw based on zone flow magnitudes

---

## 🔧 Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/newsonjacob/ReactiveOptical_Flow.git
   cd ReactiveOptical_Flow
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Place a valid AirSim `settings.json` at:
   ```
   C:\Users\<YourUsername>\Documents\AirSim\settings.json
   ```

   Example config to enable oakd-style stereo camera:
   ```json
   {
     "SettingsVersion": 1.2,
     "SimMode": "Multirotor",
     "Vehicles": {
       "UAV": {
         "VehicleType": "SimpleFlight",
         "AutoCreate": true,
         "Cameras": {
           "oakd_camera": {
             "CaptureSettings": [
               {
                 "ImageType": 0,
                 "Width": 1280,
                 "Height": 720,
                 "FOV_Degrees": 90
               }
             ],
             "X": 0.0,
             "Y": 0.0,
             "Z": -0.1,
             "Pitch": 0.0,
             "Roll": 0.0,
             "Yaw": 0.0
           }
         }
       }
     }
   }
   ```

4. Build your Unreal map and generate a packaged `.exe` (e.g. `Blocks.exe`).

5. Edit `config.ini` to point to your AirSim `settings.json` and UE4 executables. These values will be used unless overridden on the command line.

6. Run the system:
   ```bash
   python main.py
   ```
   You can override paths at runtime:
   ```bash
   python main.py --settings-path C:\path\to\settings.json --ue4-path C:\path\to\Blocks.exe
   ```
   To load a different configuration file:
   ```bash
   python main.py --config custom.ini
   ```

---

## 📊 Logs and Visualisation

- Flight logs are stored in `flow_logs/` as `.csv`
- 3D trajectory plots are saved in `analysis/` as interactive `.html` files
- Runtime messages are emitted via the standard `logging` module

---

## 📁 Folder Structure

```
ReactiveOptical_Flow/
├── main.py                   # Main control loop
├── airsim_test.py           # Test AirSim connection
├── test_oakd_camera.py      # Debug camera stream
├── uav/
│   ├── perception.py        # Optical flow tracking
│   ├── navigation.py        # Control logic
│   ├── utils.py             # Helper functions
│   └── __init__.py
├── flow_logs/               # CSV logs of motion + control
├── analysis/                # 3D path visualisations
```

---

## 🚧 Coming Soon

The current system is purely reactive. The following upgrades are planned:

- ✅ **SLAM Integration:** Add stereo ORB-SLAM2 to enable global localisation
- ✅ **FSM-Based Switching:** Use a Finite State Machine to toggle between optical flow and SLAM
- ✅ **Hybrid Navigation Mode:** Leverage both reactive and deliberative control based on context
- ⏳ **Trajectory Comparison:** Benchmark reactive vs SLAM vs hybrid strategies
- ⏳ **ROS Support (optional)**

---

## 🧠 Acknowledgements

- Built using [AirSim](https://github.com/microsoft/AirSim)
- Inspired by biologically inspired reactive navigation strategies

---

## 📬 Contact

For questions, issues, or contributions, reach out via [GitHub Issues](https://github.com/newsonjacob/ReactiveOptical_Flow/issues)
