# DME-DRL
This is the repository for "Decentralized Exploration of a Structured Environment Based on Multi-agent Deep Reinforcement Learning" : 
![demo](/res/demo.png)

# Environment Requirement
- Ubuntu 16.06 or later
- Python 3.5 or later
- Pytorch 1.4
- tensorboardX
- matplotlib
- gym

# Main Structure:
|--assets<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--`config.yaml`: configurations, robot number, communication range, etc. can be set here.<br>
|--src<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--maddapg: DME-DRL algorithm files<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--`eval_\*.py`: Evaluation codes<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--`main.py`: training code<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--`sim_utils.py`: simulator tools<br>
&nbsp;&nbsp;&nbsp;&nbsp;|--viz<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--`vis_map.py`: map visulization code<br>
     

# Installation
- Clone the repo from the github:
`git clone https://github.com/hedingjie/DME-DRL.git`
- Unzip json.tar.gz (mapset) in assets:
`tar -zvxf json.tar`;
- Generate 2D plan pictures from json files:
`python viz/vis_map.py`
- Train: `python main.py (in the src dir)`
- Evaluate: `python eval_drl.py`

# Any Questiones?
Please send email to my email: dingjiehe@gmail.com.
