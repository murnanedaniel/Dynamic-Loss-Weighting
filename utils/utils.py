import torch

def get_device():
    """
    Returns the device available to torch.
    
    Args
    ---
       None
    
    Returns
    ---
       - 'cuda' if CUDA is currently available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
