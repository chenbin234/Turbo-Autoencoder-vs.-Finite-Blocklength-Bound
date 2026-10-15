# # This file is to generatre a dataset of infomaion bits with the size of (1000000, 1, k), where k is the length of the information bits.

# import torch


# def generate_dataset_info_bits(num_samples, k, file_path="data/processed/info_bits.pt"):
#     # generate a torch tensor with the size of (num_samples, 1, k), each element is an integer number between 0 and 1
#     info_bits = torch.randint(0, 2, (num_samples, 1, k)).float()
#     print(info_bits.shape)

#     # save the tensor to a file
#     torch.save(info_bits, file_path)


# if __name__ == "__main__":

#     num_samples = 2000
#     k = 64
#     file_path = f"data/processed/train_dataset_info_bits_{num_samples}_1_{k}.pt"

#     generate_dataset_info_bits(num_samples, k, file_path)
