import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from transformers import pipeline

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration

task_prefix = "translate English to German: "
# use different length sentences to test batching
sentences = ["The house is wonderful.", "I like to work in NYC."]

# Suppose we have the following 2 training examples:
input_sequence_1 = "Welcome to NYC"
output_sequence_1 = "Bienvenue à NYC"

input_sequence_2 = "HuggingFace is a company"
output_sequence_2 = "HuggingFace est une entreprise"

# encode the inputs
task_prefix = "translate English to French: "
input_sequences = [input_sequence_1, input_sequence_2]
# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def demo_basic_t5(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # create model and move it to GPU with id rank
    model = t5_model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    input_ids = input_ids.to(rank)

    # encode the targets
    target_encoding = tokenizer(
        [output_sequence_1, output_sequence_2],
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100

    # forward pass
    loss = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    print(loss.item())

    # inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True).to(rank)

    # output_sequences = ddp_model.generate(
    #     input_ids=inputs["input_ids"],
    #     attention_mask=inputs["attention_mask"],
    #     do_sample=False,  # disable sampling to test if batching affects output
    # )

    # print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

    # training

    # loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(rank)
    # loss_fn(outputs, labels).backward()
    # optimizer.step()

    cleanup()

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def demo_model_parallel_t5(rank, world_size):
    print(f"Running DDP with model parallel t5 example on rank {rank}.")
    # setup(rank, world_size)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    # mp_model = ToyMpModel(dev0, dev1)
    device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12],
             1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

    t5_model.parallelize(device_map)
    # ddp_mp_model = DDP(t5_model)

    print('ddp model loaded')

    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    
    input_ids = input_ids.to('cuda')

    outputs = t5_model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    t5_model.deparallelize()
    # cleanup()

def demo_pipeline():
    pipe = pipeline(model="google/flan-t5-xl", device=0)
    
    for out in pipe(data()):
        print(out)

def data():
    while True:
        # This could come from a dataset, a database, a queue or HTTP request
        # in a server
        # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
        # to use multiple threads to preprocess data. You can still have 1 thread that
        # does the preprocessing while the main runs the big inference
        yield "translate English to German: The house is wonderful."

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__=="__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
        # run_demo(demo_basic, 2)
        world_size = n_gpus//2
        run_demo(demo_model_parallel_t5, world_size)
        # demo_pipeline()