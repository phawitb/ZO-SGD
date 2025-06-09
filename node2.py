import socket
import torch
import torch.nn.functional as F
from transformers import OPTForCausalLM, AutoTokenizer
from comm_utils import send_tensor, recv_tensor
from utils import params_to_vector, vector_to_params
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node2_port = 11111

mu = 0.005      # perturbation scale
eta = 5e-6      # learning rate
P = 5           # number of perturbations for gradient estimate

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
        attention_mask = attention_mask.to(hidden_states.device).to(torch.bool)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = self.final_layer_norm(hidden_states)
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

node2 = OPTNode2(layers[split_layer_idx:], full_model.model.decoder.final_layer_norm, hidden_size).to(device)
params_2 = list(node2.parameters())
x2 = params_to_vector(params_2)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', node2_port))
server.listen(1)
logger.info(f"Node2 listening on port {node2_port}")

conn, addr = server.accept()
conn.settimeout(30)  # timeout for safety
logger.info(f"Connected by {addr}")

try:
    while True:
        logger.info("Waiting to receive a_t tensor")
        a_t = recv_tensor(conn, device)
        logger.info("Received a_t tensor")

        logger.info("Waiting to receive attention_mask tensor")
        attention_mask = recv_tensor(conn, device).long()
        logger.info("Received attention_mask tensor")

        logger.info("Waiting to receive labels tensor")
        labels = recv_tensor(conn, device).long()
        logger.info("Received labels tensor")

        # Load current params
        vector_to_params(x2, params_2)

        # Compute baseline loss
        logits = node2(a_t, attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss_tensor = torch.tensor([loss.item()], device=device)
        logger.info("Sending baseline loss tensor")
        send_tensor(conn, loss_tensor)

        probs = logits.softmax(dim=-1)
        logger.info("Sending baseline probs tensor")
        send_tensor(conn, probs)

        # Estimate gradient via ZO-SGD perturbations
        g_hat_total = torch.zeros_like(x2)
        for p_idx in range(P):
            noise = torch.randn_like(x2).to(device)
            x2_pos = x2 + mu * noise
            vector_to_params(x2_pos, params_2)

            logits_plus = node2(a_t, attention_mask)
            loss_plus = F.cross_entropy(logits_plus, labels)

            loss_plus_tensor = torch.tensor([loss_plus.item()], device=device)
            logger.info(f"Sending loss_plus tensor perturbation {p_idx+1}")
            send_tensor(conn, loss_plus_tensor)

            probs_plus = logits_plus.softmax(dim=-1)
            logger.info(f"Sending probs_plus tensor perturbation {p_idx+1}")
            send_tensor(conn, probs_plus)

            g_hat = (loss_plus.item() - loss.item()) / mu
            g_hat_total += g_hat * noise

        g_hat_avg = g_hat_total / P

        # Update parameter vector
        x2 = x2 - eta * g_hat_avg
        vector_to_params(x2, params_2)
        logger.info("Updated parameters via ZO-SGD")

except (ConnectionError, RuntimeError) as e:
    logger.info(f"Connection closed or error: {e}")

conn.close()
server.close()
logger.info("Server shutdown")
