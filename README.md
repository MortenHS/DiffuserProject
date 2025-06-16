# Systematic Evaluation of Trajectory-Conditional Flow Matching for Long-Horizon Planning - Master Thesis 2025
---

This repository contains the complete code developed as part of Morten H. Sande's master's thesis for the spring of 2025. The thesis compares trajectory-conditional flow matching and diffusion for long-horizon planning tasks.

## 📌 Project Overview
- **Institution**: NTNU - Norwegian University of Science and Technology  
- **Supervisor**: Olav Egeland  
- **Co-Supervisor**: Sigmund Hennum Høeg  
- **Thesis period**: January–June 2025

**Built upon & adapted from:**
- [Diffuser (Planning with Diffusion)](https://github.com/jannerm/diffuser/tree/maze2d?tab=readme-ov-file) 
- [TCFM (Conditional Flow Matching)](https://github.com/CORE-Robotics-Lab/TCFM/tree/master)   

---

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```
---

## Config alternatives
```
config.maze2d_cfm.py [For T-CFM]
config.maze2d.py [For Diffuser]
```
## Train models
```
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
python scripts/train.py --config config.maze2d_cfm --dataset maze2d-large-v1
```

## Run planning
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
python scripts/plan_maze2d.py --config config.maze2d_cfm --dataset maze2d-large-v1
```

## Run score logging with either log_scores_optim file (planning over set amount of iterations)
```
python scripts/log_scores_optim.py --config config.maze2d --dataset maze2d-large-v1
python scripts/log_scores_optim_parallel.py --config config.maze2d_cfm --dataset maze2d-large-v1
```
