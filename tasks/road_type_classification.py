import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class LabelClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(LabelClassifier, self).__init__()
        self.net = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        :param x: (num_segments, embed_dim)
        :return:
        """
        return self.net(x)


def evaluation(city, embeddings, labels, num_fold, epochs, num_classes, device):
    print(f"--- Segment Type Classification ({city})---")
    num_segments, embed_dim = embeddings.shape
    fold_size = num_segments // num_fold

    preds = []
    scores = []
    trues = []

    for k in range(num_fold):
        fold_idx = slice(k * fold_size, (k + 1) * fold_size)
        x_val, y_val = embeddings[fold_idx], labels[fold_idx]

        left_part_idx = slice(0, k * fold_size)
        right_part_idx = slice((k + 1) * fold_size, -1)

        x_train, y_train = torch.cat([embeddings[left_part_idx], embeddings[right_part_idx]], dim=0), \
                           torch.cat([labels[left_part_idx], labels[right_part_idx]], dim=0),

        model = LabelClassifier(embed_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        best_acc = 0.
        best_pred = None
        for e in range(1, epochs + 1):
            model.train()
            pred_train = model(x_train)
            loss = criterion(pred_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            pred_score = model(x_val)
            pred_val = torch.argmax(pred_score, -1).detach().cpu()
            acc = accuracy_score(y_val.detach().cpu(), pred_val, normalize=False)
            if acc > best_acc:
                best_acc = acc
                best_pred = pred_val
                best_score = torch.softmax(pred_score, dim=1)

        preds.append(best_pred)
        scores.append(best_score.detach().cpu())
        trues.append(y_val.detach().cpu())

    preds = torch.cat(preds, dim=0)
    scores = torch.cat(scores, dim=0)
    trues = torch.cat(trues, dim=0)
    micro_f1 = f1_score(trues, preds, average='micro')
    auc = roc_auc_score(trues, scores, multi_class='ovo')
    print(f'Mi-F1: {micro_f1}, AUC: {auc}')

    return micro_f1, auc


if __name__ == '__main__':
    city_name = 'Porto'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segment_emb = np.load(f'../embeddings/segcl_{city_name}_segment_emb.npz')['data']
    segment_emb = torch.FloatTensor(segment_emb)
    segment_type_label = np.load(f'../data/{city_name}/road_network/segment_type_label.npz')['data']

    valid_indices = segment_type_label != -1
    segment_emb = segment_emb[valid_indices]
    segment_type_label = segment_type_label[valid_indices]

    evaluation(city=city_name, embeddings=segment_emb, labels=segment_type_label,
               num_fold=5, num_classes=5, epochs=100, device=device)
