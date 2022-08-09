import torch.nn as nn
import torch
import numpy as np

import vetiver


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
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
v = vetiver.VetiverModel(model = model, save_ptype = True, ptype_data=x_train,\
    model_name="torch", versioned=None, description="A regression model for testing purposes")

import pins

board = pins.board_folder(path = ".", allow_pickle_read=True)
#vetiver.vetiver_pin_write(board, v)

#vetiverapp = vetiver.VetiverAPI(v, check_ptype=True)
m = vetiver.VetiverModel.from_pin(board, "torch")
pred = m.model(torch.from_numpy(x_train))
print(pred)