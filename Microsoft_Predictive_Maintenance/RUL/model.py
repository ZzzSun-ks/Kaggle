from torch import nn


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (_, _) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.linear(last_output)
        return output
