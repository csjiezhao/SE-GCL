import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SpeedPredictor(nn.Module):
    def __init__(self, emb_dim):
        super(SpeedPredictor, self).__init__()
        self.net = nn.Linear(emb_dim, 1)

    def forward(self, x):
        return self.net(x)


def evaluation(city, embeddings, labels, num_fold, epochs, device):
    print(f"--- Segment Speed Prediction ({city}) ---")
    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        model = SpeedPredictor(embed_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_mse = 1e9
        best_pred = None
        for e in range(1, epochs + 1):
            model.train()
            pred_train = model(x_train)
            loss = criterion(pred_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            pred_val = model(x_val).detach().cpu()
            mse = mean_squared_error(y_val.detach().cpu(), pred_val)
            if mse < best_mse:
                best_mse = mse
                best_pred = pred_val
        preds.append(best_pred)
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds) ** 0.5
    print(f'MAE: {mae}, RMSE: {rmse}')
    return mae, rmse


if __name__ == '__main__':
    city_name = 'Porto'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segment_emb = np.load(f'../embeddings/segcl_{city_name}_segment_emb.npz')['data']
    segment_emb = torch.FloatTensor(segment_emb)

    segment_speed_label = np.load(f'../data/{city_name}/trajectory/segment_speed_label.npz')['data']
    segment_speed = torch.FloatTensor(segment_speed_label).unsqueeze(-1)

    evaluation(city=city_name, embeddings=segment_emb, labels=segment_speed, num_fold=5, epochs=100, device=device)
