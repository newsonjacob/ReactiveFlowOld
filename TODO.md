# ✅ TODO List – Reactive Optical Flow UAV Project

This checklist tracks planned improvements and features for the `ReactiveOptical_Flow` repository.

---

## ✅ Completed
- [x] Enable oakd_camera in AirSim via settings.json
- [x] Update main.py to auto-launch packaged UE4 sim
- [x] Fix simGetImages error by properly registering custom camera
- [x] Replace README.txt with properly formatted README.md
- [x] Confirm log and analysis output integration

---

## 🚧 In Progress
- [ ] Create FSM (`fsm.py` or `state_manager.py`) for hybrid navigation switching
- [ ] Add SLAM module integration (stereo ORB-SLAM2/3)
- [ ] Refactor hardcoded constants (camera name, image size) into `config.py` or YAML
- [ ] Improve docstrings and inline comments in all modules
- [ ] Add fallback or warning for missing UE4 executable

---

## 🔜 Planned
- [ ] Add `requirements.txt` with pinned dependencies
- [ ] Add CLI options: `--vehicle_name`, `--no-launch`, `--save`
- [ ] Move logs to `runs/` or `experiments/` for clarity
- [ ] Add `analyze.py` for log post-processing and visualisation
- [ ] Create base classes: `PerceptionModule`, `NavigationStrategy`
- [ ] Add GitHub topics and project description
- [ ] Tag current repo state as v0.1.0 (working reactive version)

---

## 📌 Notes
- This roadmap evolves with the project. Ticks off each box as you commit changes.
- Consider opening GitHub Issues for high-priority or multi-step items.
