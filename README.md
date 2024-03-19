### Setup and Requirements
Prerequisites:
- Python 3.6 or later
- PyTorch
- gymnasium
- numpy
- wandb (for logging and visualization)
- PIL (for image handling)
```
pip install torch gymnasium numpy wandb pillow moviepy imageio ipdb 
```

## Mountain Car Deterministic

### Overview
1. **mcd_dqn.py**: Implements a Deep Q-Network for the Mountain Car environment (with interventions).
2. **mcd_dqn_clean.py**: Implements a Deep Q-Network for the Mountain Car environment (without interventions).
3. **mcd_dataset.py**: Collects experiences using a provided policy and saves it as a pickle file.
4. **mcd_offlinerl.py**: Trains a new offline policy based on experiences from the pickle file collected using mcd_dataset.py.
5. **mcd_combined_offlinerl.py**: Collects experiences using a provided policy and then trains a new offline policy based on these experiences.

### Training the DQN Agent: using mcd_dqn and mcd_dqn_clean
```
python3 mc_dqn.py --name experiment_name --numeps 3000 --penalty 2 --seed 0 --wandb
```
Arguments:
- --name: Set the name of the experiment.
- --ver: Sets the version number of the experiment (useful to organize wandb runs).
- --numeps: Sets the number of episodes for training.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --seed: Sets the seed for the QNetwork.
- --wandb: Include this flag to log the training process to Weights & Biases.

### Combined Collecting Experiences and Conducting Offline RL: using mcd_combined_offlinerl
```
python mc_offlinerl.py --name experiment_name --numeps 2000 --numits 5000 --penalty 2 --policy policy_path --type intervention_type --seed 0 --wandb
```
Arguments:
- --name: Set the name of the experiment.
- --numeps: Number of episodes for collecting experiences.
- --numits: Number of iterations for offline training.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --policy: Path of the optimal policy used to collect the experiences inside the checkpoints directory.
- --type: Choose between "clean" which has no interventions, "noha" which is interventions WITHOUT human actions, and 'ha" which is interventions WITH human actions.
- --seed: Sets the seed for the Offline QNetwork.
- --wandb: Include this flag to log the training process to Weights & Biases.

### Collecting Experiences: using mcd_dataset
```
python mcd_dataset.py --name experiment_name --numeps 2000 --penalty 2 --policy policy_path --type intervention_type
```
Arguments:
- --name: Set the name of the experiment. Used as the name of the pickle file saved in the dataset folder.
- --numeps: Number of episodes for collecting experiences.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --policy: Path of the optimal policy used to collect the experiences inside the checkpoints directory.
- --type: Choose between "clean" which has no interventions, "noha" which is interventions WITHOUT human actions, and 'ha" which is interventions WITH human actions.

NOTE: Currently, you will need to modify the policy_path variable in this line of code to load the optimal policy. Notice that the policy argument only requires the path within the checkpoints directory.

### Conducting Offline RL: using mcd_offlinerl
```
python mc_offlinerl.py --name experiment_name --numits 5000  --penalty 2 --data data_path --wandb
```
Arguments:
- --name: Set the name of the experiment.
- --numits: Number of iterations for offline training.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --data: Path of the pickle file inside the dataset directory.
- --wandb: Include this flag to log the process to Weights & Biases.
