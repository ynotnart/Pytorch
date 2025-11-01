
# Sample_Housing_Prices
# Housing Price predicitons using Pytorch
# Created: 10/29/2025    
# Source: ChatGPT
# by Tony Tran

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



# Features: [square_feet, bedrooms, age_of_home]
X = np.array([
    [1200, 3, 20],
    [1500, 4, 15],
    [800, 2, 30],
    [1800, 4, 10],
    [2000, 5, 5]
], dtype=np.float32)

# Target: price in $1000s
y = np.array([[200], [250], [150], [300], [350]], dtype=np.float32)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 10),   # input layer → hidden layer
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 1)     # output layer
        )

    def forward(self, x):
        return self.layers(x)

# Create model instance
model = HousePriceModel()


criterion = nn.MSELoss()            # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")



# New house: [1600 sqft, 3 bedrooms, 8 years old]
new_house = torch.tensor([[1600, 3, 8]], dtype=torch.float32)
predicted_price = model(new_house).item()

print(f"Predicted house price: ${predicted_price*1000:.2f}")




