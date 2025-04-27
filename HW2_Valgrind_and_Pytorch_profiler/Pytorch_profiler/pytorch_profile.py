import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformer_from_scratch import Transformer

src_vocab_size = 5000
trg_vocab_size = 5000
src_pad_idx = 0
trg_pad_idx = 0

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)

batch_size = 5
src_seq_length = 10
trg_seq_length = 10

src = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_length))

device = torch.device("cpu")
model = model.to(device)
src = src.to(device)
trg = trg.to(device)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        model(src, trg)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("output.json")
