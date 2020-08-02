# DME-DRL
This is the repository for "Decentralized exploration of structured environment based on multi-agent deep reinforcement learning" : 


# Environment Requirement
- Ubuntu 16.06 or later
- Python 3.5 or later
- Pytorch 1.4
- tensorboardX
- matplotlib
- gym

# Main Structure:
|--assets
     |--config.yaml: configurations, robot number, communication range, etc. can be set here.
|--src
     |--maddapg: DME-DRL algorithm files
     |--eval_\*.py: Evaluation codes
     |--main.py: training code
     

# Installation
- Clone the repo from the github:
`git clone https://github.com/hedingjie/DME-DRL.git`
- Unzip json.tar.gz (mapset) in assets:
`tar -zvxf json.tar`;
- Generate 2D plan pictures from json files:
`python viz/vis_map.py`
- Train: `python main.py (in the src dir)`
- Evaluate: `python eval_drl.py`

# Any Questions?
Please send email to my email: dingjiehe@gmail.com.
