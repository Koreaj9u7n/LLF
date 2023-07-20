import torch
from torch_geometric.utils import to_dense_adj
import torch_geometric.utils as u
from scipy import sparse
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GATConv
class GCNNet(torch.nn.Module):
  def __init__(self,k1,k2,k3,embed_dim,num_layer,device,num_feature_xd=78,n_output=1,num_feature_xt=25,output_dim=128,dropout=0.2):
    super(GCNNet,self).__init__()
    self.device = device
    # Smile graph branch
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.embed_dim = embed_dim
    self.num_layer = num_layer
    
    # GCN
    self.Conv1 = GCNConv(num_feature_xd,num_feature_xd)
    self.Conv2 = GCNConv(num_feature_xd,num_feature_xd*2)
    self.Conv3 = GCNConv(num_feature_xd*2,num_feature_xd*4)
    
    
    # GAT
    # self.Conv1 = GATConv(num_feature_xd,num_feature_xd)
    # self.Conv2 = GATConv(num_feature_xd,num_feature_xd*2)
    # self.Conv3 = GATConv(num_feature_xd*2,num_feature_xd*4)
    
    self.relu = nn.ReLU()
    self.fc_g1 = nn.Linear(546,546*2)
    self.fc_g2 = nn.Linear(546*2,output_dim)
    self.dropout = nn.Dropout(dropout)
    
    
    self.embedding_xt = nn.Embedding(num_feature_xt+1,embed_dim)
    #protien sequence branch (LSTM)
    # self.LSTM_xt_1 = nn.LSTM(self.embed_dim,self.embed_dim,self.num_layer,batch_first = True,bidirectional=True)
    # self.fc_xt = nn.Linear(1000*256,output_dim)
    # self.fc_xt = nn.Linear(1000*256,output_dim)
    
    self.fc_sequence1=nn.Linear(384,384*2)
    self.fc_sequence2=nn.Linear(384*2,output_dim)
    
    
    
    # 1D CNN sequence
    self.protein1_cnn = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,stride=1, padding=1)
    self.protein2_cnn = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1, padding=1)
    self.protein3_cnn = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1, padding=1)
    

    #combined layers
    self.fc1 = nn.Linear(2*output_dim,1024)
    self.fc2 = nn.Linear(1024,512)
    self.out = nn.Linear(512,n_output)
  
  
  
  def sequence_cnn(self, xs, layer_cnn):
    
    if layer_cnn==1:
      xs = torch.relu(self.protein1_cnn(xs))
    
    if layer_cnn==2:
      xs = torch.relu(self.protein1_cnn(xs))
      xs = torch.relu(self.protein2_cnn(xs))
      
    if layer_cnn==3:
      xs = torch.relu(self.protein1_cnn(xs))
      xs = torch.relu(self.protein2_cnn(xs))
      xs = torch.relu(self.protein3_cnn(xs))
    
    return xs


    

  def forward(self,data,hidden,cell):
    x , edge_index, batch = data.x,data.edge_index,data.batch
    adj = to_dense_adj(edge_index)
    target = data.target
    

    if self.k1 == 1:
      h1 = self.Conv1(x,edge_index)

      h1 = self.relu(h1)
      
      h2 = self.Conv2(h1,edge_index)
  
      h2 = self.relu(h2)

      h3 = self.Conv3(h2,edge_index)

      h3 = self.relu(h3)


    # concat = torch.cat([h3,h5,h6],dim=1)
    
    concat = torch.cat([h1,h2,h3],dim=1)

    
    x = gmp(concat,batch) #global_max_pooling
    

    #flatten
    x = self.relu(self.fc_g1(x))
    x = self.dropout(x)
    x = self.fc_g2(x)
    x = self.dropout(x)
    
    
    embedded_xt = self.embedding_xt(target)
    embedded_xt=embedded_xt.permute(0, 2, 1)
    
    
    after1_1DCNN=self.sequence_cnn(embedded_xt,1)
    after2_1DCNN=self.sequence_cnn(embedded_xt,2)
    after3_1DCNN=self.sequence_cnn(embedded_xt,3)
    
    MaxPool=nn.AdaptiveMaxPool1d(1)
    
    after1_1DCNN=MaxPool(after1_1DCNN)
    after1_1DCNN=after1_1DCNN.squeeze(-1)
    
    
    after2_1DCNN=MaxPool(after2_1DCNN)
    after2_1DCNN=after2_1DCNN.squeeze(-1)
    
    after3_1DCNN=MaxPool(after3_1DCNN)
    after3_1DCNN=after3_1DCNN.squeeze(-1)

    concat = torch.cat([after1_1DCNN,after2_1DCNN,after3_1DCNN],dim=1)
    xt=self.fc_sequence1(concat)
    xt=self.fc_sequence2(xt)
    

    #concat
    xc = torch.cat((x,xt),1)
    # add some dense layers
    xc = self.fc1(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    xc = self.fc2(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    out = self.out(xc)
    
    return out
  
  def init_hidden(self, batch_size):
    hidden = torch.zeros(2,batch_size,self.embed_dim).to(self.device)
    cell = torch.zeros(2,batch_size,self.embed_dim).to(self.device)
    return hidden,cell
