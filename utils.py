import torch

def params_to_vector(params):
    vec = []
    for p in params:
        vec.append(p.data.view(-1))
    return torch.cat(vec)

def vector_to_params(vec, params):
    pointer = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(vec[pointer:pointer + numel].view_as(p))
        pointer += numel

def prepare_attention_mask(attention_mask, device):
    extended_mask = attention_mask[:, None, None, :].to(device).float()
    extended_mask = (1.0 - extended_mask) * -1e9 
    return extended_mask

def update(x, eta, g_hat, noise):
    return x - eta * g_hat * noise

def get_mem_usage():
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        mem_gb = mem.used / (1024 ** 3)
        swap_gb = swap.used / (1024 ** 3)
        return mem_gb, swap_gb
    except ImportError:
        return 0.0, 0.0
