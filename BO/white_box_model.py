import torch
import torch.nn as nn


class WhiteboxModel(nn.Module):
    def __init__(self, indexable_attributes, num_partitions, num_query, num_data_attribute, attribute_embedding_dim=4, selectivity_embedding_dim=1, frequency_embedding_dim=1,
                 size_embedding_dim=1, partition_embedding_dim=16, p_selectivity_embedding_dim=4, variance_embedding_dim=4, kurtosis_embedding_dim=4, index_embedding_dim=4, 
                 hidden_dim1=1024,hidden_dim2=512,hidden_dim3=256,hidden_dim4=128,hidden_dim5=64):
        super(WhiteboxModel, self).__init__()
        self.feature_embedding = FeatureEmbedding(indexable_attributes, num_partitions, num_data_attribute, attribute_embedding_dim, selectivity_embedding_dim, frequency_embedding_dim,
                                                  size_embedding_dim, partition_embedding_dim, p_selectivity_embedding_dim, variance_embedding_dim, kurtosis_embedding_dim, index_embedding_dim)
        self.feature_embedding_div = num_query * attribute_embedding_dim + num_query + num_query  + size_embedding_dim + partition_embedding_dim + num_partitions*p_selectivity_embedding_dim + num_partitions*variance_embedding_dim  + num_partitions*kurtosis_embedding_dim  + num_partitions * index_embedding_dim
        self.fc1 = nn.Linear(self.feature_embedding_div,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.fc6 = nn.Linear(hidden_dim5, 1)
        # Despite being written leakrelu actually uses the functionality of relu
        self.relu1 = nn.LeakyReLU(negative_slope=0.00)
        self.relu2 = nn.LeakyReLU(negative_slope=0.00)
        self.relu3 = nn.LeakyReLU(negative_slope=0.00)
        self.relu4 = nn.LeakyReLU(negative_slope=0.00)
        self.relu5 = nn.LeakyReLU(negative_slope=0.00)
        self.batchNorm1 = nn.BatchNorm1d(hidden_dim1)
        self.batchNorm2 = nn.BatchNorm1d(hidden_dim2)
        self.batchNorm3 = nn.BatchNorm1d(hidden_dim3)
        self.batchNorm4 = nn.BatchNorm1d(hidden_dim4)
        self.batchNorm5 = nn.BatchNorm1d(hidden_dim5)
        self.dropout = nn.Dropout(p = 0.25)
        self.batchNorm6 = nn.BatchNorm1d(self.feature_embedding_div)
        self.sigmoid = nn.Sigmoid()

        '''
        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) 

        # 使用 Xavier 初始化
        if isinstance(self.fc3, nn.Linear):
            nn.init.xavier_uniform_(self.fc3.weight)
            if self.fc3.bias is not None:
                nn.init.constant_(self.fc3.bias, 0.0)
        '''
        
    def forward(self, input_features):
        output = self.batchNorm6(self.feature_embedding(input_features))
        output = self.relu1(self.batchNorm1(self.fc1(output)))
        output = self.dropout(output)
        output = self.relu2((self.fc2(output)))
        output = self.dropout(output)
        output = self.relu3((self.fc3(output)))
        output = self.dropout(output)
        output = self.relu4((self.fc4(output)))
        output = self.dropout(output)
        output = self.relu5((self.fc5(output)))
        output = self.dropout(output)
        output = self.sigmoid(self.fc6(output))
        return output

class FeatureEmbedding(nn.Module):
    def __init__(self, indexable_attributes, num_partitions, num_data_attribute, attribute_embedding_dim, selectivity_embedding_dim, frequency_embedding_dim,
                 size_embedding_dim, partition_embedding_dim, p_selectivity_embedding_dim, variance_embedding_dim, kurtosis_embedding_dim, index_embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.indexable_attributes = indexable_attributes
        self.num_partitions = num_partitions
        self.attribute_embedding = nn.Linear(indexable_attributes, attribute_embedding_dim)
        self.selectivity_embedding = nn.Linear(1, selectivity_embedding_dim)
        self.frequency_embedding = nn.Linear(1, frequency_embedding_dim)
        
        self.size_embedding = nn.Linear(1, size_embedding_dim)        
        # self.partition_embedding = nn.Linear(1, partition_embedding_dim)
        self.partition_embedding = nn.Linear(num_partitions, partition_embedding_dim)
        self.p_selectivity_embedding = nn.Linear(num_data_attribute, p_selectivity_embedding_dim)
        self.variance_embedding = nn.Linear(num_data_attribute, variance_embedding_dim)
        self.kurtosis_embedding = nn.Linear(num_data_attribute, kurtosis_embedding_dim)
        
        self.index_embedding = nn.Linear(indexable_attributes, index_embedding_dim)
        
        
    def forward(self, input_features):
        batch_size = input_features[0][0].shape[0]  # 获取批量大小

        # Query features
        query_features, data_features, index_features = input_features
        attribute_embeddings = []
        selectivity_embeddings = []
        frequency_embeddings = []
        attribute_features, selectivity_features, frequency_features = query_features
        # attribute_features (batch_size, query_number, atrribute)
        attribute_embeddings = self.attribute_embedding(attribute_features.view(-1, attribute_features.shape[2]).float()).view(batch_size, -1)
        selectivity_embeddings = selectivity_features.view(batch_size, -1)
        frequency_embeddings = frequency_features.view(batch_size, -1)
        query_embedding = torch.cat([
            attribute_embeddings,
            selectivity_embeddings,
            frequency_embeddings
        ], dim=-1)

        # Data features
        size_embeddings = []
        #partition_embeddings = []
        selectivity_embeddings = []
        variance_embeddings = []
        kurtosis_embeddings = []
        data_size, p_code, data_features = data_features
        size_embeddings = self.size_embedding(data_size.view(batch_size, -1))
        p_code_onehot = []
        for i in range(batch_size):
            p = p_code[i].item()
            p = int(p * (2 ** (self.num_partitions - 1)))
            p = bin(p)[2:]
            p = p.zfill(self.num_partitions)
            p = list(map(int, p))
            p = torch.tensor(p, dtype=torch.long)
            p_code_onehot.append(p)
        p_code_onehot = torch.cat(p_code_onehot).view(batch_size, -1).to(device='cuda')
        partition_embeddings = self.partition_embedding(p_code_onehot.float())
        #partition_embeddings = p_code
        data_features = [data_features[:, :, i, :] for i in range(3)]
        p_selectivity_embeddings = self.p_selectivity_embedding(data_features[0].view(-1, data_features[0].shape[2])).view(batch_size, data_features[0].shape[1], -1)
        variance_embeddings = self.variance_embedding(data_features[1].view(-1, data_features[1].shape[2])).view(batch_size, data_features[1].shape[1], -1)
        kurtosis_embeddings = self.kurtosis_embedding(data_features[2].view(-1, data_features[2].shape[2])).view(batch_size, data_features[2].shape[1], -1)
        # p_selectivity_embeddings = self.p_selectivity_embedding(data_features[0].view(-1, data_features[0].shape[2])).view(batch_size, data_features[0].shape[1], -1)
        # variance_embeddings = data_features[1].mean(dim=-1)
        # kurtosis_embeddings = data_features[2].mean(dim=-1)
        data_embedding = torch.cat([
            size_embeddings,  # size_embeddings.mean(dim=1).unsqueeze(1),
            partition_embeddings,  # partition_embeddings.mean(dim=1).unsqueeze(1),
            # p_selectivity_embeddings.mean(dim=1),
            p_selectivity_embeddings.view(batch_size, -1),
            variance_embeddings.view(batch_size, -1),
            kurtosis_embeddings.view(batch_size, -1)
        ], dim=-1)

        # Index features
        index_embeddings = self.index_embedding(index_features.view(-1, index_features.shape[-1]).float()).view(batch_size, -1)
        # Concatenate all embeddings
        final_embedding = torch.cat([query_embedding, data_embedding, index_embeddings], dim=-1)
        return final_embedding