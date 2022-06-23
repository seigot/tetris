import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=4, stride=2,padding=1,
                padding_mode='zeros',bias=False),
                nn.ReLU())
        
        self.conv2 = nn.Sequential(
                nn.ConstantPad2d((2,2,2,2),0),
                nn.Conv2d(32, 32, kernel_size=5, stride=1,padding=0,
                bias=False),
                nn.ReLU())
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=2, 
                bias=False, padding_mode='zeros'),
                nn.ReLU())
        self.num_feature = 64*4*1
        self.fc1 = nn.Sequential(nn.Linear(self.num_feature,256))
        self.fc2 = nn.Sequential(nn.Linear(256,256))
        self.fc3 = nn.Sequential(nn.Linear(256,1))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)    
        x = x.view(-1, self.num_feature )
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x