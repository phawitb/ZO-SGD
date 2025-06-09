import socket
import torch
from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from comm_utils import send_tensor, recv_tensor
from utils import params_to_vector, vector_to_params, prepare_attention_mask
import logging
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
max_length = 256
mu = 0.005
eta = 5e-6
P = 5
seed_base = 42
node2_host = '192.168.1.45'  # เปลี่ยนเป็น IP ของ node2 จริง
node2_port = 11111

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
full_model = OPTForCausalLM.from_pretrained("facebook/opt-125m").to(device)
layers = list(full_model.model.decoder.layers)
split_layer_idx = 6

class OPTNode1(torch.nn.Module):
    def __init__(self, embed_tokens, layers, final_layer_norm):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = torch.nn.ModuleList(layers)
        self.final_layer_norm = final_layer_norm

    def forward(self, input_ids, attention_mask):
        attention_mask = attention_mask.to(input_ids.device)
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        h = self.final_layer_norm(h)
        return h

node1 = OPTNode1(full_model.model.decoder.embed_tokens,
                 layers[:split_layer_idx],
                 full_model.model.decoder.final_layer_norm).to(device)

params_1 = list(node1.parameters())
x1 = params_to_vector(params_1)

dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"].select(range(100))

def preprocess_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)

train_dataset = train_dataset.map(preprocess_fn, batched=True)

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
logger.info(f"Connecting to Node2 at {node2_host}:{node2_port}...")
sock.connect((node2_host, node2_port))
logger.info(f"Connected to Node2 at {node2_host}:{node2_port}")

train_log = []

for epoch in range(20):
    total_correct = 0
    total_samples = 0
    loss_accum = 0.0
    grad_scalar_accum = 0.0
    batch_count = 0
    comm_cost_bytes = 0

    epoch_start = time.time()

    for t, batch in enumerate(train_loader):
        batch_start = time.time()
        torch.manual_seed(seed_base + epoch * len(train_loader) + t)

        vector_to_params(x1, params_1)
        input_ids = batch["input_ids"].to(device)
        attention_mask = prepare_attention_mask(batch["attention_mask"], device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            a_t = node1(input_ids, attention_mask)

        comm_cost_bytes += tensor_bytes(a_t)
        logger.info(f"Sending a_t tensor batch {t+1}")
        send_tensor(sock, a_t)

        comm_cost_bytes += tensor_bytes(attention_mask)
        logger.info(f"Sending attention_mask tensor batch {t+1}")
        send_tensor(sock, attention_mask)

        comm_cost_bytes += tensor_bytes(labels.float())
        logger.info(f"Sending labels tensor batch {t+1}")
        send_tensor(sock, labels.float())

        logger.info(f"Receiving loss tensor batch {t+1}")
        loss_tensor = recv_tensor(sock, device)
        comm_cost_bytes += tensor_bytes(loss_tensor)
        baseline_loss = loss_tensor.item()

        logger.info(f"Receiving probs tensor batch {t+1}")
        probs = recv_tensor(sock, device)
        comm_cost_bytes += tensor_bytes(probs)
        preds = torch.argmax(probs, dim=-1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        g_hat_total = torch.zeros_like(x1)

        for p_idx in range(P):
            noise = torch.randn_like(x1).to(device)
            x1_pos = x1 + mu * noise
            vector_to_params(x1_pos, params_1)

            with torch.no_grad():
                a_t_plus = node1(input_ids, attention_mask)

            comm_cost_bytes += tensor_bytes(a_t_plus)
            logger.info(f"Sending a_t_plus tensor batch {t+1} perturbation {p_idx+1}")
            send_tensor(sock, a_t_plus)

            comm_cost_bytes += tensor_bytes(attention_mask)
            logger.info(f"Sending attention_mask tensor batch {t+1} perturbation {p_idx+1}")
            send_tensor(sock, attention_mask)

            comm_cost_bytes += tensor_bytes(labels.float())
            logger.info(f"Sending labels tensor batch {t+1} perturbation {p_idx+1}")
            send_tensor(sock, labels.float())

            logger.info(f"Receiving loss_plus tensor batch {t+1} perturbation {p_idx+1}")
            loss_plus_tensor = recv_tensor(sock, device)
            comm_cost_bytes += tensor_bytes(loss_plus_tensor)
            loss_plus = loss_plus_tensor.item()

            logger.info(f"Receiving probs_plus tensor batch {t+1} perturbation {p_idx+1}")
            probs_plus = recv_tensor(sock, device)
            comm_cost_bytes += tensor_bytes(probs_plus)

            g_hat = (loss_plus - baseline_loss) / mu
            g_hat_total += g_hat * noise

        g_hat_avg = g_hat_total / P
        x1 = x1 - eta * g_hat_avg

        loss_accum += baseline_loss
        grad_scalar_accum += g_hat_avg.mean().item()
        batch_count += 1

        batch_end = time.time()
        logger.info(
            f"Epoch {epoch+1} Batch {t+1} done: Loss={baseline_loss:.4f}, "
            f"Grad_scalar={g_hat_avg.mean().item():.6f}, "
            f"Accuracy={(correct / labels.size(0)):.4f}, "
            f"Time={batch_end - batch_start:.2f}s, "
            f"CommCost={comm_cost_bytes / 1e6:.3f} MB"
        )

    epoch_end = time.time()
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
    epoch_loss = loss_accum / batch_count if batch_count > 0 else 0.0
    epoch_grad_scalar = grad_scalar_accum / batch_count if batch_count > 0 else 0.0
    epoch_comm_mb = comm_cost_bytes / 1e6

    logger.info(
        f"Epoch {epoch+1} summary: Accuracy={epoch_acc:.4f}, Loss={epoch_loss:.4f}, "
        f"Grad_scalar={epoch_grad_scalar:.6f}, Time={epoch_end - epoch_start:.2f}s, "
        f"CommCost={epoch_comm_mb:.3f} MB"
    )

    train_log.append(
        {
            "Epoch": epoch + 1,
            "Accuracy": epoch_acc,
            "Loss": epoch_loss,
            "Grad_scalar": epoch_grad_scalar,
            "Time_sec": epoch_end - epoch_start,
            "CommCost_MB": epoch_comm_mb,
            "Algorithm": "ZO-SGD TCP (No Backprop)",
        }
    )

    df_log = pd.DataFrame(train_log)
    df_log.to_csv("train_log_summary.csv", index=False)
    logger.info("Saved train_log_summary.csv")

sock.close()
logger.info("Connection closed")
