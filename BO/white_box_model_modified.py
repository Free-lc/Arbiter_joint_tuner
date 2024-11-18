import torch
import torch.nn as nn


class WhiteboxModel(nn.Module):
    def __init__(self, indexable_attributes, num_partitions, num_query, num_data_attribute, attribute_embedding_dim=1, selectivity_embedding_dim=1, frequency_embedding_dim=1,
                 size_embedding_dim=1, partition_embedding_dim=128, p_selectivity_embedding_dim=1, variance_embedding_dim=1, kurtosis_embedding_dim=1, index_embedding_dim=1, 
                 hidden_dim1=256,hidden_dim2=128,hidden_dim3=64,hidden_dim4=32):
        super(WhiteboxModel, self).__init__()
        self.feature_embedding = FeatureEmbedding(indexable_attributes, num_partitions, num_data_attribute, attribute_embedding_dim, selectivity_embedding_dim, frequency_embedding_dim,
                                                  size_embedding_dim, partition_embedding_dim, p_selectivity_embedding_dim, variance_embedding_dim, kurtosis_embedding_dim, index_embedding_dim)
        self.feature_embedding_div =  size_embedding_dim + num_partitions + partition_embedding_dim*p_selectivity_embedding_dim + partition_embedding_dim*variance_embedding_dim  + partition_embedding_dim*kurtosis_embedding_dim  + num_partitions * index_embedding_dim
        self.fc1 = nn.Linear(self.feature_embedding_div,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, 1)

        # Despite being written leakrelu actually uses the functionality of relu
        self.relu = nn.LeakyReLU(negative_slope=0.00)
        self.dropout = nn.Dropout(p = 0.25)#        output = self.dropout(output),放在relu后面，sigmoid后面不要放
        self.batchNorm = nn.BatchNorm1d(self.feature_embedding_div)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        output = self.batchNorm(self.feature_embedding(input_features))
        output = self.relu((self.fc1(output)))
        output = self.dropout(output)
        output = self.relu((self.fc2(output)))
        output = self.dropout(output)
        output = self.relu((self.fc3(output)))
        output = self.dropout(output)
        output = self.relu((self.fc4(output)))
        output = self.dropout(output)
        output = self.sigmoid(self.fc5(output))
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

        self.partition_embedding = nn.Linear(num_partitions, partition_embedding_dim)
        self.p_selectivity_embedding1 = nn.Linear(num_data_attribute, p_selectivity_embedding_dim)
        self.variance_embedding1 = nn.Linear(num_data_attribute, variance_embedding_dim)
        self.kurtosis_embedding1 = nn.Linear(num_data_attribute, kurtosis_embedding_dim)
        self.p_selectivity_embedding2 = nn.Linear(num_partitions, partition_embedding_dim)
        self.variance_embedding2 = nn.Linear(num_partitions, partition_embedding_dim)
        self.kurtosis_embedding2 = nn.Linear(num_partitions, partition_embedding_dim)
        
        self.index_embedding1 = nn.Linear(indexable_attributes, index_embedding_dim)
        self.index_embedding2 = nn.Linear(num_partitions, partition_embedding_dim)
        
        
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
        size_embeddings = data_size.view(batch_size, -1)

        p_code_onehot = []
        for i in range(batch_size):
            p = p_code[i][0].tolist()
            p = int(''.join(map(str, p)))
            p = bin(p)[2:]
            p = p.zfill(self.num_partitions)
            p = list(map(int, p))
            p = torch.tensor(p, dtype=torch.long)
            p_code_onehot.append(p)
        p_code_onehot = torch.cat(p_code_onehot).view(batch_size, -1).to(device='cuda')
        partition_embeddings = self.partition_embedding(p_code_onehot.float())#partition_embeddings = p_code
    
        data_features = [data_features[:, :, i, :] for i in range(3)]
        p_selectivity_embeddings = self.p_selectivity_embedding1(data_features[0].view(-1, data_features[0].shape[2])).view(batch_size, data_features[0].shape[1], -1)
        variance_embeddings      = self.variance_embedding1(data_features[1].view(-1, data_features[1].shape[2])).view(batch_size, data_features[1].shape[1], -1)
        kurtosis_embeddings      = self.kurtosis_embedding1(data_features[2].view(-1, data_features[2].shape[2])).view(batch_size, data_features[2].shape[1], -1)
        p_selectivity_embeddings = self.p_selectivity_embedding2(p_selectivity_embeddings.view(batch_size, -1))
        variance_embeddings      = self.variance_embedding2(variance_embeddings.view(batch_size, -1))
        kurtosis_embeddings      = self.kurtosis_embedding2(kurtosis_embeddings.view(batch_size, -1))

        data_embedding = torch.cat([
            size_embeddings, 
            p_code_onehot, 
            p_selectivity_embeddings,
            variance_embeddings,
            kurtosis_embeddings
        ], dim=-1)

        # Index features
        index_embeddings = self.index_embedding1(index_features.view(-1, index_features.shape[-1]).float()).view(batch_size, -1)
        #index_embeddings = self.index_embedding2(index_embeddings)
        # Concatenate all embeddings
        final_embedding = torch.cat([data_embedding,index_embeddings], dim=-1)#query_embedding, 
        return final_embedding