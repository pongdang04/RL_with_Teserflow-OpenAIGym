import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN:
    def __init__(self, input_size, output_size, name="main", learning_rate=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        
        # Neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def predict(self, state):
        state = np.reshape(state, [1, self.input_size])
        x = torch.FloatTensor(state)
        with torch.no_grad():
            return self.model(x).numpy()
    
    def update(self, x_stack, y_stack):
        x = torch.FloatTensor(x_stack)
        y = torch.FloatTensor(y_stack)
        
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), None
    
    def copy_from(self, other):
        """Copy weights from another DQN"""
        self.model.load_state_dict(other.model.state_dict())