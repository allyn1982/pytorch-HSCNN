import numpy as np
import pandas as pd
import torch
from model import HSCNN
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from data_loader import Resizer, LungDataset
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import  roc_curve, auc
from scipy import interp
from utils import roc_curves, class_weight, classify_image, cv_data, weights_init, plot_loss


# read in data
seed = 1

all_data = pd.read_csv('path')
record_id = all_data['path'].tolist()
label4 = all_data['label4'].tolist()

skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

best_acc = 0
best_acc_list = []

best_acc_size = 0
best_acc_list_size = []

best_acc_consistency = 0
best_acc_list_consistency = []

best_acc_margin = 0
best_acc_list_margin = []

for index, (train_indices, test_indices) in enumerate(skf.split(record_id, label4)):
    train_path, val_path = cv_data(train_indices, test_indices, record_id, label4)

    # training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    network = HSCNN(num_tasks=3, num_class_list=[3, 2, 2]).to(device)
    network.apply(weights_init)

    train_data = LungDataset(train_path, num_tasks=3, transform=transforms.Compose([Resizer()]))
    val_data = LungDataset(val_path, transform=transforms.Compose([Resizer()]))

    train_loader = DataLoader(train_data, shuffle=True, num_workers=4, batch_size=6, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=True, num_workers=4, batch_size=1, drop_last=True)

    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    train_size = pd.read_csv(train_path, header=None).shape[0]
    val_size = pd.read_csv(val_path, header=None).shape[0]
    dataset_sizes = {'train': train_size, 'val': val_size}

    sub_task_weights = torch.tensor([0.33, 0.34, 0.33])

    class_weight_dict = class_weight(train_path, num_tasks=3)

    optim1 = optim.Adam(network.parameters(), lr=1e-3)  # ,momentum=.9)

    optimizer_ft = optim1

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    model_ft1, train_loss, val_loss, train_acc, val_acc, best_fold_acc, best_fold_acc_size, best_fold_acc_consistency, best_fold_acc_margin = train_model(
        network,
        optimizer_ft,
        exp_lr_scheduler,
        sub_task_weights,
        class_weight_dict,
        best_acc,
        num_tasks=3,
        num_epochs=100)

    best_acc_list.append(best_fold_acc)
    best_acc_list_size.append(best_fold_acc_size)
    best_acc_list_consistency.append(best_fold_acc_consistency)
    best_acc_list_margin.append(best_fold_acc_margin)

    torch.save(model_ft1.state_dict(), 'path')

    plot_loss(train_loss, val_loss, train_acc, val_acc)

    # load saved model
    network = HSCNN(num_tasks=3, num_class_list=[3, 2, 2]).to(device)
    network.load_state_dict(torch.load('path'))

    # evaluate the model
    tp, tn, fp, fn, true_label_list, score_list = classify_image(network, val_path)

    # compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(true_label_list, score_list)

    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

#     # save the model w/ the lowest loss across all folds
#     best_acc = best_fold_acc

print('--------------Malignancy---------------')
roc_curves(tprs, mean_fpr, aucs)
print('Aucs:', aucs)

print('Mean Accuracy:', np.mean(best_acc_list))
print('Accuracy:', best_acc_list)

print('--------------Size---------------')
print('Size Acc:', best_acc_list_size)
print('Size Mean Acc:', np.mean(best_acc_list_size))

print('--------------Consistency---------------')
print('Consistency Acc:', best_acc_list_consistency)
print('Consistency Mean Acc:', np.mean(best_acc_list_consistency))

print('--------------Margin---------------')
print('Margin Acc:', best_acc_list_margin)
print('Margin Mean Acc:', np.mean(best_acc_list_margin))




