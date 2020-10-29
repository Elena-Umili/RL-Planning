import torch
device = 'cpu'
code_size = 5
z = torch.randn(code_size)
print(z)
z = z.where(z < 0.0, torch.zeros(code_size).to(device), torch.ones(code_size).to(device))
print(z)
