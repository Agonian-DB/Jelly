import torch

# 创建一个需要梯度的矩阵 A
A = torch.randn(3, 3, requires_grad=True)

# 对 A 进行 SVD 分解
U, S, V = torch.linalg.svd(A)

# 定义一个基于奇异值的损失函数（例如，奇异值之和）
loss = S.sum()

# 反向传播，计算梯度
loss.backward()

# 查看 A 的梯度
print("梯度 dL/dA:")
print(A.grad)
