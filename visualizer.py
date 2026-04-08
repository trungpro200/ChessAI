import torchshow as ts 
import torch

tens: torch.Tensor = torch.load("debug.ts")

views = [tens[0][:, x].view(8,8) for x in range(12, 36)]
print(views[-1])
ts.show(views,nrows = 4, ncols = 4, mode="grayscale", auto_permute=False)