import socket
import torch
import torch.nn.functional as F
from transformers import OPTForCausalLM, AutoTokenizer
from comm_utils import send_tensor, recv_tensor
from utils import params_to_vector, vector_to_params, update
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eta = 5e-6
node2_port = 11111

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
full_model = OPTForCausalLM.from_pretrained("facebook/opt-125m").to(device)
layers = list(full_model.model.decoder.layers)
split_layer_idx = 6
hidden_size = full_model.config.hidden_size

class OPTNode2(torch.nn.Module):
    def __init__(self, layers, final_layer_norm, hidden_size, num_labels=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.final_layer_norm = final_layer_norm
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask.to(hidden_states.device).to(torch.bool)  # แก้ตรงนี้
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = self.final_layer_norm(hidden_states)
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def log_recv(tensor_name, tensor):
    shape = tensor.shape
    size_bytes = tensor_bytes(tensor)
    logger.info(f"Recv {tensor_name} shape={shape}, size={size_bytes / 1e6:.3f} MB")
    return size_bytes

def log_send(tensor_name, tensor):
    shape = tensor.shape
    size_bytes = tensor_bytes(tensor)
    logger.info(f"Send {tensor_name} shape={shape}, size={size_bytes / 1e6:.3f} MB")
    return size_bytes

node2 = OPTNode2(layers[split_layer_idx:], full_model.model.decoder.final_layer_norm, hidden_size).to(device)
params_2 = list(node2.parameters())
x2 = params_to_vector(params_2)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', node2_port))
server.listen(1)
logger.info(f"Node2 listening on port {node2_port}")

conn, addr = server.accept()
logger.info(f"Connected by {addr}")

try:
    while True:
        a_t = recv_tensor(conn, device)
        log_recv("a_t", a_t)

        attention_mask = recv_tensor(conn, device).long()
        log_recv("attention_mask", attention_mask)

        labels = recv_tensor(conn, device).long()
        log_recv("labels", labels)

        vector_to_params(x2, params_2)

        a_t.requires_grad_()

        logits = node2(a_t, attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss_tensor = torch.tensor([loss.item()], device=device)
        log_send("loss_tensor", loss_tensor)
        send_tensor(conn, loss_tensor)

        probs = logits.softmax(dim=-1)
        log_send("probs", probs)
        send_tensor(conn, probs)  

        loss.backward()
        grad_vector = torch.cat([p.grad.flatten() for p in params_2 if p.grad is not None])
        x2 = x2 - eta * grad_vector
        vector_to_params(x2, params_2)

except ConnectionError:
    logger.info("Connection closed by Node1")

conn.close()
server.close()
logger.info("Server shutdown")
