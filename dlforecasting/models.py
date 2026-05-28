import torch.nn as nn


class HybridModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=1,
        model_type='LSTM'
    ):
        super().__init__()

        self.model_type = model_type

        if model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        else:
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out.squeeze()