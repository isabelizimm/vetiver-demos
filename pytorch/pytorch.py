import torch.nn as nn
import numpy as np

from vetiver import VetiverModel, VetiverAPI

from vetiver.handlers._interface import create_translator


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# # Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
# for epoch in range(num_epochs):
#     # Convert numpy arrays to torch tensors
#     inputs = torch.from_numpy(x_train)
#     targets = torch.from_numpy(y_train)

#     # Forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
    
#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
v = VetiverModel(model = model, save_ptype = True, ptype_data=x_train,\
    model_name="my_model", versioned=None, description="A regression model for testing purposes")

vetiverapp = VetiverAPI(v, check_ptype=True)

vetiverapp.run()