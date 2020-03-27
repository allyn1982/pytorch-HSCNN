
import torch
import torch.nn as nn
import torch.nn.functional as F


class HSCNN(nn.Module):
    """
    This is a Pytorch version of the HSCNN used in the paper https://doi.org/10.1117/12.2551220  
    
    Input: num_tasks
    Output: HSCNN model
    
    num_tasks: This is to specify the total number of semantic tasks of the model. 
                In the paper mentioned above, there are three semantic tasks, which are
                mean diameter, consistency, and margin of a pulmonary nodule.
                Please also modify the number of self.subtask and self.task
                modules if this num_tasks is to be modified.
                
    Note: The input dimensions of the image patch is 52x52x52 voxels, which is the same 
          as in the original HSCNN paper https://arxiv.org/pdf/1806.00712.pdf
    """
    def __init__(self, num_tasks=3):   
        super(HSCNN, self).__init__()
        
        self.num_tasks = num_tasks
        
        # feature module
        self.feature_module = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)), 
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))  
        )

        self.dropout1 = nn.Dropout3d(p=0.6, inplace=False)  

        # low level tasks    
        self.subtask_1 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.BatchNorm1d(64),
            nn.Linear(64, 3), # 3 classes for semantic task
            nn.Softmax(dim=1)
        )
        
        self.subtask_2 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(), 
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.BatchNorm1d(64),
            nn.Linear(64, 2), # 2 classes for semantic task
            nn.Softmax(dim=1)
        )
        
        self.subtask_3 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(), 
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2), # 2 classes for semantic task
            nn.Softmax(dim=1)
        )
        
        self.task_1 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False)
        )
        
        self.task_2 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False)
        )
        
        self.task_3 = nn.Sequential(
            nn.Linear(32*13*13*13, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.8, inplace=False)
        )

        # high level task module
        self.dense3 = nn.Linear((32*13*13*13+256*self.num_tasks), 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dense_output2 = nn.Linear(256, 2) # 2 classes for malignancy task
        
    def forward(self, x):
        # feature module 1. input layer
        x = x

        #feature module 
        out = self.feature_module(x)
        out = out.view(-1, 32*13*13*13) #flatten
        out = self.dropout1(out)
        
        # make lists to save outputs
        output_list = []
        out_cat_list = []
        out_cat_list.append(out)
        
        # low level tasks
        out_low_1 = self.subtask_1(out)
        out_low_2 = self.subtask_2(out)
        out_low_3 = self.subtask_3(out)
        output_list.append(out_low_1)
        output_list.append(out_low_2)
        output_list.append(out_low_3)       

        # high level task
        out_high_1 = self.task_1(out)
        out_high_2 = self.task_2(out)
        out_high_3 = self.task_3(out)
        out_cat_list.append(out_high_1)
        out_cat_list.append(out_high_2)
        out_cat_list.append(out_high_3)

        out_high = torch.cat(out_cat_list, dim=1) #col-wise
        out_high = self.dense3(out_high)
        out_high = nn.ReLU()(out_high)
        out_high = self.bn7(out_high)
        out_high = self.dense_output2(out_high)
        out_high = F.softmax(out_high, dim=1)
        output_list.append(out_high)
        return output_list
    
# print out the network
model = HSCNN()
print(model)

# check with random input
data =torch.rand(1,1,52,52,52).uniform_(0, 255)
model.eval()
model(data)