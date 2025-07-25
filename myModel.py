import torch
from torch import nn

INPUT_SIZE=8
HIDDEN_NEURONS1=8
HIDDEN_NEURONS2=4

class myModel(nn.Module):
  def __init__(self):
    super(myModel, self).__init__()
    self.InputLayer=nn.Linear(INPUT_SIZE, HIDDEN_NEURONS1)
    self.HiddenLayer1=nn.Linear(HIDDEN_NEURONS1, HIDDEN_NEURONS2)
    self.relu=nn.ReLU()
    self.HiddenLayer2=nn.Linear(HIDDEN_NEURONS2, 1)
    self.OutputLayer=nn.Sigmoid()

  def forward(self, x):
    x=self.InputLayer(x)
    x=self.HiddenLayer1(x)
    x=self.relu(x)
    x=self.HiddenLayer2(x)
    x=self.OutputLayer(x)
    return x