import struct
import torch
import numpy as np

def send_all(sock, data):
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken during send")
        total_sent += sent

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise RuntimeError("Socket connection broken during recv")
        data += packet
    return data

def send_tensor(sock, tensor):
    shape = torch.tensor(tensor.shape, dtype=torch.int64)
    shape_bytes = struct.pack(f'{len(shape)}q', *shape.tolist())
    send_all(sock, struct.pack('Q', len(shape))) 
    send_all(sock, shape_bytes)  

    dtype_str = str(tensor.dtype)
    dtype_bytes = dtype_str.encode('utf-8')
    send_all(sock, struct.pack('Q', len(dtype_bytes))) 
    send_all(sock, dtype_bytes)  

    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    send_all(sock, struct.pack('Q', len(tensor_bytes)))  
    send_all(sock, tensor_bytes)

def recv_tensor(sock, device):
    shape_len = struct.unpack('Q', recv_all(sock, 8))[0]
    shape_bytes = recv_all(sock, shape_len * 8)
    shape = struct.unpack(f'{shape_len}q', shape_bytes)

    dtype_len = struct.unpack('Q', recv_all(sock, 8))[0]
    dtype_bytes = recv_all(sock, dtype_len)
    dtype_str = dtype_bytes.decode('utf-8')
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]  

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "uint8": torch.uint8,
    }

    np_dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int64": np.int64,
        "int32": np.int32,
        "uint8": np.uint8,
    }

    dtype = dtype_map.get(dtype_str)
    np_dtype = np_dtype_map.get(dtype_str)

    if dtype is None or np_dtype is None:
        raise ValueError(f"Unsupported dtype received: {dtype_str}")

    tensor_len = struct.unpack('Q', recv_all(sock, 8))[0]
    tensor_bytes = recv_all(sock, tensor_len)

    np_array = np.frombuffer(tensor_bytes, dtype=np_dtype)
    tensor = torch.from_numpy(np_array).reshape(shape).to(device)
    return tensor
