from torch import nn
from torch.nn import functional as func


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(64 * 7 * 7, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)

  def forward(self, x):
    x = self.pool(func.relu(self.conv1(x)))
    x = self.pool(func.relu(self.conv2(x)))
    x = x.view(-1, 64 * 7 * 7)
    x = func.relu(self.fc1(x))
    x = func.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = CNN()