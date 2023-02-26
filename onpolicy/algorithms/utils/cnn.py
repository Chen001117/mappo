import torch
import torch.nn as nn

class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size
        active_func = [nn.Tanh(), nn.ReLU()][self._use_ReLU]

        channel = obs_shape[0] # 2
        input_width = obs_shape[1] # 29
        input_height = obs_shape[2] # 29
        assert input_width==29 and input_height==29

        self.cnn = nn.Sequential(
            nn.Conv2d(channel,self.hidden_size//4,3,2,1),
            nn.BatchNorm2d(self.hidden_size//4),
            active_func,
            nn.Conv2d(self.hidden_size//4,self.hidden_size//2,3,2,1),
            nn.BatchNorm2d(self.hidden_size//2),
            active_func,
            nn.Conv2d(self.hidden_size//2,self.hidden_size,3,2,1),
            nn.BatchNorm2d(self.hidden_size),
            active_func,
            nn.Conv2d(self.hidden_size,self.hidden_size,4,1,0),
            nn.Flatten()
            # active_func,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x




# import torch.nn as nn
# from .util import init

# """CNN Modules and utils."""

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)


# class CNNLayer(nn.Module):
#     def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1, padding=1):
#         super(CNNLayer, self).__init__()

#         active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
#         gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

#         # def init_(m):
#         #     return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

#         input_channel = obs_shape[0]
#         input_width = obs_shape[1]
#         input_height = obs_shape[2]

#         self.cnn = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=input_channel,
#                 out_channels=hidden_size // 2,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding
#             ),
#             active_func,
#             nn.MaxPool2d(kernel_size, stride=kernel_size),
#             nn.Conv2d(
#                 in_channels=hidden_size // 2,
#                 out_channels=hidden_size // 2,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding
#             ),
#             active_func,
#             Flatten(),
#             nn.Linear(
#                 hidden_size // 2 * (input_width//kernel_size) * (input_height//kernel_size),
#                 hidden_size
#             ),
#             active_func,
#             nn.Linear(hidden_size, hidden_size), 
#             active_func
#         )

#     def forward(self, x):
#         # x = x / 255.0
#         x = self.cnn(x)
#         return x


# class CNNBase(nn.Module):
#     def __init__(self, args, obs_shape):
#         super(CNNBase, self).__init__()

#         self._use_orthogonal = args.use_orthogonal
#         self._use_ReLU = args.use_ReLU
#         self.hidden_size = args.hidden_size

#         self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

#     def forward(self, x):
#         x = self.cnn(x)
#         return x
