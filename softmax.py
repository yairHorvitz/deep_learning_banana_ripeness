import torch
from torch import nn


# Define the softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Single linear layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along class dimension

    def forward(self, x):
        logits = self.linear(x)
        probabilities = self.softmax(logits)  # Apply softmax to logits
        return probabilities


# def softmax(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
#     """
#     Apply softmax to a tensor of logits along a specified dimension.
#     Args:
#         logits (torch.Tensor): Input tensor (e.g., [batch_size, num_classes]).
#         dim (int): Dimension along which to apply softmax. Defaults to -1 (last dimension).
#     Returns:
#         torch.Tensor: Output tensor with softmax applied along the specified dimension.
#     """
#     # Subtract max along the specified dimension for numerical stability
#     logits_stable = logits - torch.max(logits, dim=dim, keepdim=True).values
#     exp_logits = torch.exp(logits_stable)
#     return exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)


def main():
    # Define the input and output dimensions
    input_dim = 224 * 224 * 3  # all pictures are 224x224 pixels with 3 channels(RGB)
    output_dim = 4  # Number of classes


    # Create a model instance
    model = SoftmaxClassifier(input_dim, output_dim)
    print(model)

if main == "__main__":
    main()
