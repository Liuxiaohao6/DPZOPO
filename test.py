import torch
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
b = torch.tensor([[False, True, False], [False, False, True], [False, False, True]]).to_sparse()
print(b.indices())
indices = [*b.indices()]
print(indices[0].shape[0])
z = torch.normal(mean=0, std=1, size=(indices[-1].shape[0],), dtype=a.dtype)
print(z)
print(a[indices])