"""Build the PyTorch dataset for the channel model."""

import torch
from torch.utils.data import Dataset


class InfobitDataset(Dataset[torch.Tensor]):
    """
    A PyTorch dataset for information bits of length k.

    This dataset loads the information bits (which are randomly generated)
    and returns them as a PyTorch tensor.

    Attributes:
        device: The device on which tensors will be loaded.
        data: The tensor containing the information bits.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the sample at the specified index.

    """

    def __init__(self, num_samples: int, k: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate a tensor of binary values with shape (num_samples, k, 1).
        self.data = torch.randint(low=0, high=2, size=(int(num_samples), k, 1)).float()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.

        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieve a specific item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            torch.tensor: The information bits at the specified index.

        """
        return self.data[index, :, :].to(self.device)
