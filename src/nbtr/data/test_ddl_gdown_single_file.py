from nbtr.data.ddl_gdown import DistributedDataLoader as DDL
from nbtr.train.trainer import VAL_DATA_FILES 
from nbtr.data.data_common import write_datafile
import shutil
import torch
import glob
import os
import time

B=4
T=5
N_PROC=4
N_ITER=3
N_SHARDS=1
SHARD_SIZE = N_ITER * N_PROC * B * T
DATA = list(range(N_SHARDS*SHARD_SIZE+1))
TEST_OUTPUT_DIR="test_output_train_single"
def mock_download_wrapper():
    shard_index = -1
    def _mock_download_function(id, filename, output_dir):
        download_files=sorted(glob.glob(f"{output_dir}/*.bin"))
        if len(download_files)>2:
            file_to_delete = download_files[1]
            os.remove(file_to_delete)
            print(f"{file_to_delete} deleted!")
            
        nonlocal shard_index
        shard_index+=1
        
        print(f"Downloading {id} to {output_dir}/{filename}")
        write_datafile(f"{output_dir}/{filename}", DATA[shard_index*SHARD_SIZE:(shard_index+1)*SHARD_SIZE+1])
        
    return _mock_download_function

def reference_batches_x(rank):
    return [torch.tensor(DATA[(rank*B*T+i*N_PROC*B*T):(rank*B*T+i*N_PROC*B*T)+(B*T)], dtype=torch.int64).view(B,T) for i in range(N_ITER*N_SHARDS)]
    
def test_load_data_rank_0():
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    rank = 0
    local_rank=0
    ddl = DDL(VAL_DATA_FILES, TEST_OUTPUT_DIR, B, T, rank, local_rank, N_PROC, mock_download_wrapper())
    batches = reference_batches_x(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    
    # simulate work
    time.sleep(1)
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])


def test_load_data_rank_1():
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    rank = 1
    local_rank=0
    ddl = DDL(VAL_DATA_FILES, TEST_OUTPUT_DIR, B, T, rank, local_rank, N_PROC, mock_download_wrapper())
    batches = reference_batches_x(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])

    # simulate work
    time.sleep(1)
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    

def test_load_data_rank_2():
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    rank = 2
    local_rank=0
    ddl = DDL(VAL_DATA_FILES, TEST_OUTPUT_DIR, B, T, rank, local_rank, N_PROC, mock_download_wrapper())
    batches = reference_batches_x(rank)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # sanity check on x,y data    
    assert torch.all(x[0][1:]==y[0][:-1])
    assert torch.all(x[-1][1:]==y[-1][:-1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])

    # simulate work
    time.sleep(1)
    
    # next shard (shard-1)
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[0])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[1])
    
    # simulate work
    time.sleep(1)
    
    x,y = ddl.next_batch()
    assert torch.equal(x, batches[2])
    
if __name__=='__main__':
    test_load_data_rank_0()