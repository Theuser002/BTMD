import torch
torch.__version__

# def softXEnt (input, target):
#     logprobs = torch.nn.functional.log_softmax (input, dim = 1)
#     return  -(target * logprobs).sum() / input.shape[0]

class softXEnt ():
    def __init__ (self):
        self
        
    def __call__ (self, input, target):
        print(input.shape, target.shape)
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0] 