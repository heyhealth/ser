import torch
from torch.nn import functional as F

log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,reduction='sum')
# loss.backward()

print(loss)
