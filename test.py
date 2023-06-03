import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model and an optimizer
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Print the initial value of the model parameter
print(model.linear.weight)

# Create a dummy input and compute the loss
x = torch.tensor([[1.0]])
y_true = torch.tensor([[2.0]])
loss_fn = nn.MSELoss()
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

# Compute the gradients and update the model parameter
loss.backward()
optimizer.step()

# Print the updated value of the model parameter
print(model.linear.weight)
