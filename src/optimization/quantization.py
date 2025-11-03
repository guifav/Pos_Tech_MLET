
import torch.nn as nn
import torch.quantization as quantization


def quantize(values, levels):
    """
    Quantizes the input values to the specified number of levels.

    :param values: List of floating-point numbers to be quantized.
    :param levels: Number of quantization levels.
    :return: List of quantized values.
    """
    min_val = min(values)
    max_val = max(values)
    step = (max_val - min_val) / (levels - 1)
    
    quantized_values = []
    for value in values:
        quantized_value = round((value - min_val) / step) * step + min_val
        quantized_values.append(quantized_value)
    
    return quantized_values


# Define a simple neural network for demonstration purposes
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 1000)
        self.fc3 = nn.Linear(1000, 5000)
        self.fc4 = nn.Linear(5000, 2500)
        self.fc5 = nn.Linear(2500, 1250)
        self.fc6 = nn.Linear(1250, 625)
        self.fc7 = nn.Linear(625, 312)
        self.fc8 = nn.Linear(312, 156)
        self.fc9 = nn.Linear(156, 78)
        self.fc10 = nn.Linear(78, 39)
        self.fc11 = nn.Linear(39, 19)
        self.fc12 = nn.Linear(19, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the neural network and print its structure
model = SimpleNN()
print("Original model:", model)

# Prepare the model for quantization
model.qconfig = quantization.default_qconfig
quantization.prepare(model, inplace=True)

# Calibrate the model with some dummy data
dummy_input = torch.randn(10, 10)
model(dummy_input)

# Convert the model to a quantized version
quantization.convert(model, inplace=True)
print("Quantized model:", model)

# Save the quantized model
torch.save(model.state_dict(), 'quantized_model.pth')