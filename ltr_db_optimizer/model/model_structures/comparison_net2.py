import torch.nn 

from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees

# This code snippet was mostly extracted, with some renaming from: 
# https://github.com/learnedsystems/BaoForPostgreSQL/blob/master/bao_server/net.py
class LTRComparisonNet(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRComparisonNet, self).__init__() 
        self.input_dimension_1 = in_dim_1 # Dimension of the Tree Convolution Layers
        self.input_dimension_2 = in_dim_2 # Dimension of the Query Encoding
        
        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1+16, 512), 
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()), 
            BinaryTreeConv(512, 256),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )
        
        self.comparison_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1+16, 256), 
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()), 
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()), 
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
        )
        
        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,16),
            torch.nn.LeakyReLU()
        ) 
        
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(16+32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,1)
        )

    #def get_input_channels(self):
    #    return self.input_channels
    
    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query)
        objects = self.object_net(trees)#.repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)        
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums/(comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums),1) 
        return self.output_net(with_query_enc)
        
    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        #query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query)
        objects = self.object_net(trees)#.repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees) 
        #print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons 
        comparison_sums = comparison_sums/(comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums),1) 
        return self.output_net(with_query_enc)
    