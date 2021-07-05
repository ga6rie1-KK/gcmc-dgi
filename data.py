import torch

class DataSet:
    def __init__(self):
        self.adj_matrix = torch.tensor([[1,0],[0,2],[0,3]])
        self.num_rating = 3
        self.u_sidefeat = torch.randn(3,5)
        self.v_sidefeat = torch.randn(2,5)
