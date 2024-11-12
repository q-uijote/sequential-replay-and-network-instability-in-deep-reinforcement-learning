# Sequential Replay Simulations using Torch

## Overview

A torch version of the class 'SFMA_DQNAgent' used in the experiments of https://github.com/sencheng/EM_Spatial_Learning.git (**Modeling the function of episodic memory in spatial learning (Zeng, X., Diekmann, N., Wiskott, L., Cheng, S., 2023)**) is implemented in deep_sfma_torch.py.

Similarly, sequential_replay_1_torch.py is ment to reproduce the results of the respective 'sequential_replay_1.py' program in the directory mentioned above.

## Prerequisites

The file structure for correct execution looks like this:

The three directories mentioned in installation steps 1-3 are placed in the same folder. 

## Installation

### 1. Clone the repository
```

```
### 2. Clone the original EM_Spatial_Learning directory
```
git clone https://github.com/sencheng/EM_Spatial_Learning.git 
```
Line 15 and 16 must be deleted in EM_Spatial_Learning/interfaces/oai_gym_interfaces.py in order to match the requirements.txt file.

### 3. Download the latest version of CoBeL-RL 

CoBeL_RL_nd_new_features

### 4. Create a python environment with the required packages listed in requirements.txt
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=/path/to/your/repo/CoBeL_RL_nd_new_features:/path/to/your/repo/EM_Spatial_Learning
```
### 5. Run the experiment
```
python sequential_replay_1_torch.py
```
The data will be stored in data/sequential_replay_1_torch

