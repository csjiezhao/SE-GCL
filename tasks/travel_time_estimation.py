import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TravelTimeEstimator(nn.Module):
    def __init__(self, segment_embeddings, num_segments, embed_dim, hidden_dim, num_layers, out_dim):
        super(TravelTimeEstimator, self).__init__()
        self.emb_lookup = nn.Embedding.from_pretrained(segment_embeddings, freeze=True, padding_idx=num_segments)
        self.encoder = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_len):
        """
        :param x: (batch_size, seq_len)
        :param x_len: (batch_size, )
        :return y: (batch_size, out_dim)
        """
        x_emb = self.emb_lookup(x)  # (batch, seq_len, emb_dim)
        x_emb = x_emb.permute(1, 0, 2)  # fit the input format of RNN (i.e., no batch-first).
        x_emb = pack_padded_sequence(x_emb, x_len, enforce_sorted=False)
        state, hn = self.encoder(x_emb)  # (seq_len, batch, hidden_dim), (1, batch, hidden_dim)
        y = self.decoder(hn[-1])
        return y


def evaluation(city, task_data_path, embeddings, epochs, device):
    print(f"\n--- Travel Time Estimation ({city}) ---")

    X = np.load(os.path.join(task_data_path, 'time_est_x.npz'))['data']
    X_len = np.load(os.path.join(task_data_path, 'time_est_x_len.npz'))['data']
    Y = np.load(os.path.join(task_data_path, 'time_est_y.npz'))['data']

    X = torch.LongTensor(X)
    X_len = torch.IntTensor(X_len)
    Y = torch.FloatTensor(Y).unsqueeze(-1)

    split = int(X.shape[0] * 0.8)
    X_train, X_test = X[:split], X[split:]
    X_len_train, X_len_test = X_len[:split], X_len[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    train_set = torch.utils.data.TensorDataset(X_train, X_len_train, Y_train)
    test_set = torch.utils.data.TensorDataset(X_test, X_len_test, Y_test)

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    num_segments, embed_dim = embeddings.shape[0] - 1, embeddings.shape[1]
    model = TravelTimeEstimator(segment_embeddings=embeddings, num_segments=num_segments, embed_dim=embed_dim,
                                hidden_dim=embed_dim, num_layers=2, out_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss().to(device)

    best = [0, 1e9, 1e9]  # best epoch, best mae, best rmse
    for e in range(1, epochs + 1):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            x, x_len, y = batch_data
            x, x_len, y = x.to(device), x_len, y.to(device)
            y_pred = model(x, x_len)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        trues = []
        preds = []
        for batch_idx, batch_data in enumerate(test_loader):
            x, x_len, y = batch_data
            x, x_len, y = x.to(device), x_len, y.to(device)
            trues.append(y.cpu())
            preds.append(model(x, x_len).detach().cpu())
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        mae = mean_absolute_error(trues, preds)
        rmse = mean_squared_error(trues, preds) ** 0.5
        print(f'Epoch: {e}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        if mae < best[1]:
            best = [e, mae, rmse]
    print(f'Best epoch: {best[0]}, MAE: {best[1]:.4f}, RMSE: {best[2]:.4f}')
    return best[1], best[2]


if __name__ == '__main__':
    city_name = 'Porto'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segment_emb = np.load(f'../embeddings/segcl_{city_name}_segment_emb.npz')['data']
    segment_emb = torch.FloatTensor(segment_emb)
    pad_emb = torch.full((1, segment_emb.shape[1]), fill_value=-1)
    segment_emb = torch.cat([segment_emb, pad_emb], dim=0).to(device)

    evaluation(task_data_path=f'../data/{city_name}/trajectory/', embeddings=segment_emb, epochs=100, device=device)
