import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import sys
sys.path.append('/data2/fray/index_selection_evaluation')
from BO.train_loader import MyDataset
from BO.white_box_model_modified import WhiteboxModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# 定义训练的设备 使用gpu进行训练
device = torch.device("cuda")

query_features = []
data_features = []
index_features = [] 

#BCELoss with weight
class BCELoss_with_weight(nn.Module):
    def __init__(self):
 
        super().__init__()
 
    def forward(self, outputs, batch_labels, weight):
        bceloss = nn.BCELoss(reduction="none")
        loss_without_weight = bceloss(outputs,batch_labels)
        loss_with_weight = loss_without_weight * weight
        loss = torch.mean(loss_with_weight)
        return loss
    
#load input
sample_input = []
objectives_list = []
sample_num = 0
with open('BO/train/tpch100.pkl', 'rb') as f1, open('BO/train/tpch100_label.pkl', 'rb') as f2:
    while True:
        try:
            query_features, data_features, index_features = pickle.load(f1)
            query_features_device  = []
            data_features_device = []
             
            for query_feature in query_features:
                query_features_device.append(query_feature.to(device))
            for data_feature in data_features:
                data_features_device.append(data_feature.to(device))
            index_features_device = index_features.to(device)
            objectives, constraints = pickle.load(f2)

            input_list = []
            input_list.append(query_features_device)
            input_list.append(data_features_device)
            input_list.append(index_features_device)
            input_list.extend(objectives)
            input_list.extend([0])

            sample_input.append(input_list)
            objectives_list.extend(objectives)
    
            sample_num += 1
            if sample_num == 1280:
                break
        except EOFError:
            break
max_len = max([sample_input[i][1][1].shape[1] for i in range(len(sample_input))])
for i in range(len(sample_input)):
    if sample_input[i][1][1].shape[1] < max_len:
        padded_tensor = F.pad(sample_input[i][1][1], (max_len-sample_input[i][1][1].shape[1], 0))
        sample_input[i][1][1] = padded_tensor

#sort objectives and tag sample_input
objectives_list = sorted(objectives_list)
flag = objectives_list[int(len(objectives_list)*0.5)]
print('flag:',flag)
for input in sample_input:
    input[-1] = 1 if input[-2] < flag else 0

    
train_data = [input[0:3] for input in sample_input]
labels = [input[-1] for input in sample_input]

#train data
batch_size = 128
train_dataset = MyDataset(train_data[0:1152], labels[0:1152])
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test data
test_dataset1 = MyDataset(train_data[1152:1280], labels[1152:1280])
test_dataset2 = MyDataset(train_data[0:1152], labels[0:1152])
testloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, shuffle=True)
testloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)
#net and loss_function and optimizer 
indexable_attributes = len(query_features[0][0])    #2
num_partitions = len(index_features)    #250
num_query = len(query_feature)   #3
num_data_attribute = len(data_feature[0][0])

whiteboxmodels = WhiteboxModel(indexable_attributes = indexable_attributes, num_partitions = num_partitions, num_query=num_query,num_data_attribute=num_data_attribute)
whiteboxmodels = whiteboxmodels.to(device)
loss_function = BCELoss_with_weight()
loss_function = loss_function.to(device)

learning_rate = 0.01
optimizer = torch.optim.Adam(whiteboxmodels.parameters(),lr = learning_rate, weight_decay = 0.01)#Adadelta  #if (epoch+1)%5 == 0:optimizer.param_groups[0]['lr'] /= 10 , lr = learning_rate
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#train
num_epochs = 200
lr_reduce_counter = 0
last_lr = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    last_lr = optimizer.param_groups[0]['lr']
    for batch_features, batch_labels in dataloader:
        # transform batch_labels to correct shape and type
        batch_labels = batch_labels.float().view(-1, 1)
        batch_labels = batch_labels.to(device)
        # forward
        outputs = whiteboxmodels(batch_features)
        # compute weight
        weight = []
        for batch_laber in batch_labels:
            a = batch_laber.tolist()
            if a[0] == 0.0:
                weight.extend([1])
            elif a[0] ==  1.0:
                weight.extend([3])
        weight = torch.tensor(weight)
        weight = weight.view(-1, 1)
        weight = weight.to(device)
        # compute loss
        loss = loss_function(outputs = outputs, batch_labels = batch_labels, weight = weight)
        
        #加正则
        regularization_loss = 0
        for parameter in whiteboxmodels.parameters():
            regularization_loss += torch.sum(torch.abs(parameter))
        loss += 0.0001 * regularization_loss
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    scheduler.step(running_loss)
    if last_lr != optimizer.param_groups[0]['lr']:
        lr_reduce_counter += 1
    if lr_reduce_counter == 4:
        break
    #print('[epoch: %d ]  loss: %.3f'%(epoch+1,running_loss))

#Set the save path for the whiteboxmodels
whiteboxmodels_PATH ='./whiteboxmodels.pth'
torch.save(whiteboxmodels.state_dict(), whiteboxmodels_PATH)

#load whiteboxmodes
PATH ='./whiteboxmodels.pth'
net = WhiteboxModel(indexable_attributes = indexable_attributes, num_partitions = num_partitions, num_query=num_query,num_data_attribute=num_data_attribute)
net.load_state_dict(torch.load(PATH))
net = net.to(device)

#test
correct0 = 0
total0 = 0
correct1 = 0
total1 = 0
correct = 0
total = 0
white_output = []
for batch_features, batch_labels in testloader1:
    batch_labels = batch_labels.float().view(-1, 1)
    batch_labels = batch_labels.to(device)
    outputs = net(batch_features)
    for batch_laber,output in zip(batch_labels,outputs):
        batch_label = batch_laber.tolist()
        output = output.tolist()
        white_output.append(float(output[0] >= 0.5))
        if batch_laber[0] == 0.0 and float(output[0] >= 0.5) == batch_laber[0]:
            correct0 += 1
            total0 += 1
        elif batch_laber[0] == 1.0 and float(output[0] >= 0.5) == batch_laber[0]:
            correct1 += 1
            total1 += 1
        elif batch_laber[0] == 0.0:
            total0 += 1
        elif batch_laber[0] == 1.0:
            total1 += 1
try:
    with open('visualization/bar/whitebox_scores.pkl', 'wb') as f:
        pickle.dump(white_output,f)
        print(white_output)
except (FileNotFoundError, EOFError):
    print("QAQ")
correct = correct0 + correct1
total = total0 + total1
accuracy = correct / total
accuracy0 = correct0 / total0
accuracy1 = correct1 / total1
print('测试集总准确率:', accuracy)
print('测试集标签为0的准确率:', accuracy0)
print('测试集标签为1的准确率:', accuracy1)
print('---------------------------------')

correct0 = 0
total0 = 0
correct1 = 0
total1 = 0
correct = 0
total = 0
for batch_features, batch_labels in testloader2:
    batch_labels = batch_labels.float().view(-1, 1)
    batch_labels = batch_labels.to(device)
    outputs = net(batch_features)
    for batch_laber,output in zip(batch_labels,outputs):
        batch_label = batch_laber.tolist()
        output = output.tolist()
        if batch_laber[0] == 0.0 and float(output[0] >= 0.5) == batch_laber[0]:
            correct0 += 1
            total0 += 1
        elif batch_laber[0] == 1.0 and float(output[0] >= 0.5) == batch_laber[0]:
            correct1 += 1
            total1 += 1
        elif batch_laber[0] == 0.0:
            total0 += 1
        elif batch_laber[0] == 1.0:
            total1 += 1
            
correct = correct0 + correct1
total = total0 + total1
accuracy = correct / total
accuracy0 = correct0 / total0
accuracy1 = correct1 / total1
print('训练集总准确率:', accuracy)
print('训练集标签为0的准确率:', accuracy0)
print('训练集标签为1的准确率:', accuracy1)
