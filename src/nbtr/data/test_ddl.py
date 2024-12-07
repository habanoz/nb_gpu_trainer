from nbtr.data.ddl import DistributedDataLoader
import os
from nbtr.data.data_common import write_datafile
import torch
import pdb

B=4
T=5
N_PROC=4
N_ITER=3
N_SHARDS=2
SHARD_SIZE = N_ITER * N_PROC * B * T
DATA = list(range(N_SHARDS*SHARD_SIZE+1))

def generate_test_data():
    os.makedirs("test_data_dir", exist_ok=True)
    
    for i in range(N_SHARDS):
        write_datafile(f"test_data_dir/test_train_00000{i}.bin", DATA[i*SHARD_SIZE:(i+1)*SHARD_SIZE])

def reference_batches(rank):
    return [torch.tensor(DATA[(rank*B*T+i*N_PROC*B*T):(rank*B*T+i*N_PROC*B*T)+(B*T)], dtype=torch.int64).view(B,T) for i in range(N_ITER*N_SHARDS)]
    
def test_load_data_rank_0():
    rank = 0
    ddl = DistributedDataLoader(filename_pattern="test_data_dir/*_train_*.bin", B=B, T=T, process_rank=rank, num_processes=N_PROC )
    batches = reference_batches(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[3])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[4])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[5])


def test_load_data_rank_1():
    rank = 1
    ddl = DistributedDataLoader(filename_pattern="test_data_dir/*_train_*.bin", B=B, T=T, process_rank=rank, num_processes=N_PROC )
    batches = reference_batches(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])

    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[3])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[4])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[5])
    
def test_load_data_rank_2():
    rank = 2
    ddl = DistributedDataLoader(filename_pattern="test_data_dir/*_train_*.bin", B=B, T=T, process_rank=rank, num_processes=N_PROC )
    batches = reference_batches(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[3])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[4])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[5])
    
    
def test_load_data_rank_3():
    rank = 3
    ddl = DistributedDataLoader(filename_pattern="test_data_dir/*_train_*.bin", B=B, T=T, process_rank=rank, num_processes=N_PROC )
    batches = reference_batches(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[3])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[4])
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[5])
    print(x)
    print(y)
    print(DATA)
    
if __name__ == '__main__':
    test_load_data_rank_1()