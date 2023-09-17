import torch
from tqdm import tqdm
import h5py
import numpy as np

class GraspDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.keys = self.get_keys()
        self.size = len(self.keys)

    def get_keys(self):
        with h5py.File(self.hdf5_path, "r") as dataset:
            keys = []
            for k in dataset:
                keys.append(k)
            return keys
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            group = dataset.get(self.keys[index])

            observations = np.array(group['observations']).astype(np.float64)
            actions = np.array(group['actions']).astype(np.float64)
            rewards = np.array(group['rewards']).astype(np.float64)
            terminals = np.array(group['terminals']).astype(np.uint8)
            next_observations = np.array(group['next_observations'].astype(np.float64))

            # print("self._observations[indices],: ", observations.shape, type(observations), type(observations[0]))
            # print("self._actions[indices],: ", actions.shape, type(actions[0]))
            # print("self._rewards[indices],: ", rewards.shape, type(rewards[0]))
            # print("self._next_obs[indices],: ", next_observations.shape, type(next_observations[0]))
            # print("self._terminals[indices],: ", terminals.shape, type(terminals[0]))

            return dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                next_observations=next_observations
            )
            # return (obs, action, reward)