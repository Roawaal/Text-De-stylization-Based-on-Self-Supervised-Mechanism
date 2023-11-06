import torch

if __name__ == '__main__':
    check = torch.cuda.is_available()
    print(check)