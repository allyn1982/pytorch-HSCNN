import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from data_loader import Resizer, LungDataset
from torch.utils.data import DataLoader
from sklearn.metrics import auc

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def class_weight(train_path, num_tasks=3):
    df = pd.read_csv(train_path, header=None)
    label_list = []
    for i in range(num_tasks + 1):
        label_list.append(df.iloc[:, i].tolist())

    class_weight_dict = {}

    for i in range(len(label_list)):
        labels = label_list[i]
        num_classes = len(np.unique(labels))

        weight_list = []

        for j in range(num_classes):
            count = float(labels.count(int(j)))
            weight = 1 / (count / float(len(labels)))
            weight_list.append(weight)

        class_weight_dict[i] = torch.FloatTensor(weight_list).cuda()

    return class_weight_dict

def train_model(train_path, dataset_sizes, model, optimizer, scheduler, sub_task_weights, class_weight_dict, best_acc, dataloaders_dict, device, num_tasks=3, num_epochs=0):
    best_model_wts = copy.deepcopy(model.state_dict())

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            running_loss_list = [0.0] * (num_tasks + 1)

            label_corrects_list = [0.0] * (num_tasks + 1)

            # Iterate over data.
            for batch in dataloaders_dict[phase]:

                image, labels = batch['img'].to(device), batch['label'].to(device)

                labels = labels[:, 0]
                #                 print(labels)

                true_label_list = []
                for i in range(num_tasks + 1):
                    true_label_list.append(labels[:, i].long())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = model(image)
                    #                     if phase == 'val':
                    #                         print(outputs)

                    class_weight_dict = class_weight(train_path, num_tasks=3)

                    loss_list = []

                    for i in range(num_tasks + 1):
                        criterion_weighted = nn.CrossEntropyLoss()  # pass weights to all tasks
                        loss_list.append(criterion_weighted(outputs[i], true_label_list[i]))

                    # backward + optimize only if in training phase
                    if phase == 'train':

                        sub_task_weights = sub_task_weights.to(device)

                        loss = 0
                        for i in range(num_tasks):
                            loss += loss_list[i] * sub_task_weights[i]

                        loss = (loss / num_tasks) + loss_list[-1]

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * image.size(0)
                for i in range(num_tasks + 1):
                    running_loss_list[i] += loss_list[i].item() * image.size(0)
                    label_corrects_list[i] += outputs[i].argmax(dim=1).eq(true_label_list[i]).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_loss_list = []
            label_acc_list = []
            for i in range(num_tasks + 1):
                epoch_loss_list.append(running_loss_list[i] / dataset_sizes[phase])
                label_acc_list.append(label_corrects_list[i] / dataset_sizes[phase])

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(label_acc_list[-1])
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(label_acc_list[-1])

            # print loss
            string = '{} total loss: {:.4f} '
            args = [phase, epoch_loss]
            for i in range(num_tasks + 1):
                string += 'label' + str(i + 1) + '_loss: {:.4f} '
                args.append(epoch_loss_list[i])

            print(string.format(*args))

            # print accuracy
            string_2 = '{} '
            args_2 = [phase]
            for i in range(num_tasks + 1):
                string_2 += 'label' + str(i + 1) + '_Acc: {:.4f} '
                args_2.append(label_acc_list[i])

            print(string_2.format(*args_2))

            # deep copy the model
            if phase == 'val' and label_acc_list[-1] > best_acc:
                print('saving with acc of {}'.format(label_acc_list[-1]),
                      'improved over previous {}'.format(best_acc))
                best_acc = label_acc_list[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc_size = label_acc_list[0]
                best_acc_consistency = label_acc_list[1]
                best_acc_margin = label_acc_list[2]

    print('Best val acc: {:4f}'.format(float(best_acc)))

    best_fold_acc = best_acc
    best_fold_acc_size = best_acc_size
    best_fold_acc_consistency = best_acc_consistency
    best_fold_acc_margin = best_acc_margin

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss, val_loss, train_acc, val_acc, best_fold_acc, best_fold_acc_size, best_fold_acc_consistency, best_fold_acc_margin


def plot_loss(train_loss, val_loss, train_acc, val_acc):
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.plot(train_acc, label='train acc')
    plt.plot(val_acc, label='val acc')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    return plt.show()


def classify_image(model, val_path, train_indices, test_indices,record_id, device):
    # set parameters to 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label_list = []
    score_list = []

    # load image
    test_path = cv_data(train_indices, test_indices, record_id, record_id)[-1]
    val_data = LungDataset(val_path, transform=transforms.Compose([Resizer()]))
    val_loader = DataLoader(val_data, shuffle=True, num_workers=4, batch_size=1)

    model.eval()

    for batch in val_loader:

        image, labels = batch['img'].to(device), batch['label'].to(device)

        labels = labels[:, -1]

        true_label = int(labels[-1][-1].cpu().numpy())  # select label 4
        true_label_list.append(true_label)
        #         print(true_label)

        with torch.no_grad():

            # make prediction\n",
            pred = model(image)
            score = pred[-1].tolist()[0][true_label]
            #         print(score)

            #         print(pred[-1])
            score_list.append(score)
            pred_label = pred[-1].argmax(dim=1)
            #         print(pred_label)

            # make classification
            if pred_label > 0.5:
                pred_label = 1
            else:
                pred_label = 0

            # update parameters
            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 0 and true_label == 0:
                tn += 1
            elif pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 1:
                fn += 1
            else:
                print('ERROR')

    print('tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn)

    return tp, tn, fp, fn, true_label_list, score_list


def cv_data(train_indices, test_indices, record_id, all_data):
    train_path_list = []
    train_label_list_1 = []
    train_label_list_2 = []
    train_label_list_3 = []
    train_label_list_4 = []

    test_path_list = []
    test_label_list_1 = []
    test_label_list_2 = []
    test_label_list_3 = []
    test_label_list_4 = []

    for i in range(len(train_indices)):
        train_patch_path_original = record_id[train_indices[i]]
        train_patch_path_flipped_x = train_patch_path_original.split('.')[0] + '_flipped_x.npy'
        train_patch_path_flipped_y = train_patch_path_original.split('.')[0] + '_flipped_y.npy'
        train_patch_path_flipped_z = train_patch_path_original.split('.')[0] + '_flipped_z.npy'

        train_path_list.append(train_patch_path_original)
        train_label_list_1.append(int(all_data['label1'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_2.append(int(all_data['label2'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_3.append(int(all_data['label3'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_4.append(int(all_data['label4'][all_data['path'] == record_id[train_indices[i]]].values))

        # add paths and labels of augmented patches to csv
        train_path_list.append(train_patch_path_flipped_x)
        train_label_list_1.append(int(all_data['label1'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_2.append(int(all_data['label2'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_3.append(int(all_data['label3'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_4.append(int(all_data['label4'][all_data['path'] == record_id[train_indices[i]]].values))

        train_path_list.append(train_patch_path_flipped_y)
        train_label_list_1.append(int(all_data['label1'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_2.append(int(all_data['label2'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_3.append(int(all_data['label3'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_4.append(int(all_data['label4'][all_data['path'] == record_id[train_indices[i]]].values))

        train_path_list.append(train_patch_path_flipped_z)
        train_label_list_1.append(int(all_data['label1'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_2.append(int(all_data['label2'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_3.append(int(all_data['label3'][all_data['path'] == record_id[train_indices[i]]].values))
        train_label_list_4.append(int(all_data['label4'][all_data['path'] == record_id[train_indices[i]]].values))

    for i in range(len(test_indices)):
        test_path_list.append(record_id[test_indices[i]])
        test_label_list_1.append(int(all_data['label1'][all_data['path'] == record_id[test_indices[i]]].values))
        test_label_list_2.append(int(all_data['label2'][all_data['path'] == record_id[test_indices[i]]].values))
        test_label_list_3.append(int(all_data['label3'][all_data['path'] == record_id[test_indices[i]]].values))
        test_label_list_4.append(int(all_data['label4'][all_data['path'] == record_id[test_indices[i]]].values))

    df_train = pd.DataFrame({'path': train_path_list,
                             'label1': train_label_list_1,
                             'label2': train_label_list_2,
                             'label3': train_label_list_3,
                             'label4': train_label_list_4})

    df_test = pd.DataFrame({'path': test_path_list,
                            'label1': test_label_list_1,
                            'label2': test_label_list_2,
                            'label3': test_label_list_3,
                            'label4': test_label_list_4})

    df_train.to_csv('path', header=False, index=False)
    df_test.to_csv('path', header=False, index=False)

    train_path = 'path'
    val_path = 'path'

    return train_path, val_path

def roc_curves(tprs, mean_fpr, aucs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr,
             mean_tpr,
             color='y',
             label='Mean ROC HSCNN (AUC = %0.2f $\\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2,
             alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    return plt.show()