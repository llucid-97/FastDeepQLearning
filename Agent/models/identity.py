from torch import nn
class Identity(nn.Module):
    def __init__(self,*args,**kwargs):
        super(Identity, self).__init__()

    def forward(self,*args):
        return args if len(args)>1 else args[0]