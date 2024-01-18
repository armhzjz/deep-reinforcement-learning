import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    networks = []  #  saves a list of all the identifiers used to create networks

    def __init__(self, state_size, action_size, identifier: str, seed=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        if identifier in QNetwork.networks:
            raise Exception(
                f'The type of network "{identifier}" already has been declared - {QNetwork.networks}'
            )

        super(QNetwork, self).__init__()
        if seed:
            self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.identifier = identifier  # identifies if network is target / local
        QNetwork.networks.append(identifier)
        self.fc_main = nn.Linear(8, 512)
        self.fc_lateral = nn.Linear(8, 512)
        self.fc_physics = nn.Linear(8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def __del__(self):
        """ Before the instance is destroyed, remove its declared network type"""
        QNetwork.networks.remove(self.identifier)

    @property
    def nettype(self):
        return self.identifier

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #o = self.relu(self.conv_hidden1(state))
        #o = self.relu(self.conv_hidden2(o))
        #o = self.relu(self.conv_hidden3(o))
        #o = self.relu(self.fc1(o.view(-1, 64*7*7)))
        o_main = self.relu(self.fc_main(state))
        o_lateral = self.relu(self.fc_lateral(state))
        o_physics = self.relu(self.fc_physics(state))
        o = self.relu(self.fc2(o_main + o_lateral + o_physics))
        return self.fc3(o)
