import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # convolutional layer sees 84 x 84 x 4 ( state's size should be C x H x W )
        #self.conv_hidden1 = nn.Conv2d(4,32,8,4)
        # convolutional layer sees 20 x 20 x 32 ( state's size should be C x H x W )
        #self.conv_hidden2 = nn.Conv2d(32,64,4,2)
        # convolutional layer sees 9 x 9 x 64 ( state's size should be C x H x W )
        #self.conv_hidden3 = nn.Conv2d(64,64,3,1)
        # linear layer 7 x 7 x 64 -> 512
        #self.fc1 = nn.Linear(64*7*7, 512)
        # linear layer 512 -> action_size
        #self.fc2 = nn.Linear(512, action_size)
        #self.relu = nn.ReLU()
        self.fc1 = nn.Linear(8,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #o = self.relu(self.conv_hidden1(state))
        #o = self.relu(self.conv_hidden2(o))
        #o = self.relu(self.conv_hidden3(o))
        #o = self.relu(self.fc1(o.view(-1, 64*7*7)))
        o = self.relu(self.fc1(state))
        o = self.relu(self.fc2(o))
        return self.fc3(o)

