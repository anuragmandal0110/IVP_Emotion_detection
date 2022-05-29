import torch

class PoseClassification(torch.nn.Module):

    def __init__(self):
        super(PoseClassification, self).__init__()
        self.linear1 = torch.nn.Linear(1000, 500)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(500, 500)
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(500, 100)
        self.output = torch.nn.Linear(100, 2)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
