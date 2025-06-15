# 🧠 Systematic Evaluation of Trajectory-Conditional Flow Matching for Long-Horizon Planning  
---

## 📌 Project Attribution

**Authors**  
- *Morten Husby Sande*

**Supervisors**  
- *Olav Egeland*  
- *Sigmund Hennum Høeg*

**Built upon & adapted from:**
- [TCFM (Conditional Flow Matching)](https://github.com/CORE-Robotics-Lab/TCFM/tree/master)   
- [Diffuser (Planning with Diffusion)](https://github.com/jannerm/diffuser/tree/maze2d?tab=readme-ov-file) 

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
