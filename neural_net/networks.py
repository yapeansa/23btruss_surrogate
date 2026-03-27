from torch import nn

class Net3(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, depth, init=0.1):
        super().__init__()

        self.input = nn.Linear(n_feature, n_hidden)
        self.layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(depth)])
        self.predict = nn.Linear(n_hidden, n_output)

        self._init_weights(init)

        self.activation = nn.SiLU()

    def _init_weights(self, init):
        nn.init.xavier_uniform_(self.input.weight, gain=init)
        nn.init.ones_(self.input.bias)

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=init)
            nn.init.ones_(layer.bias)

        nn.init.xavier_uniform_(self.predict.weight, gain=init)
        nn.init.ones_(self.predict.bias)
    
    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)
        return x
