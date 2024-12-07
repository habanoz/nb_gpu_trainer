# modified from
# https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
##############
import numpy as np
import glob
import torch
import gdown
import os
import glob
import threading
import time

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

def _download_data(gd_id, filename, output_dir):
    download_files=sorted(glob.glob(f"{output_dir}/*.bin"))
    if len(download_files)>2:
        file_to_delete = download_files[1]
        os.remove(file_to_delete)
        print(f"{file_to_delete} deleted!")
    gdown.download(id=gd_id, output=f"{output_dir}/{filename}")

def _download_data_with_retry(gd_id, filename, output_dir, download_function, max_retry=5, wait_before_retry=30):
    
    fn_download = download_function if download_function else _download_data
    
    for i in range(max_retry):
        try:
            fn_download(gd_id, filename, output_dir)
            break
        except Exception as e:
            print(f"Downloading file '{filename}' failed with error '{e}', trial {i}")
            time.sleep(wait_before_retry)
            
def _download_file_in_background(gd_id, filename, output_dir, download_function=None):
    thread = threading.Thread(target=_download_data_with_retry, args=(gd_id, filename, output_dir, download_function))
    thread.daemon = True
    thread.start()

class DistributedDataLoader:
    def __init__(self, file_names_dict, output_dir, B, T, process_rank, local_process_rank, num_processes, fn_mock_download=None):
        self.process_rank = process_rank
        self.local_process_rank = local_process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = list(file_names_dict.keys())
        self.output_dir = output_dir
        self.file_names_dict = file_names_dict
        self.fn_mock_download = fn_mock_download
        
        assert len(self.files) > 0, f"did not find any files for name:{output_dir}"

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        if not os.path.exists(f"{self.output_dir}/{self.files[0]}") and self.local_process_rank==0:
            os.makedirs(self.output_dir, exist_ok=True)
            
            print("Downloading", f"{self.output_dir}/{self.files[0]}")
            gd_id = self.file_names_dict[self.files[0]]
            _download_data_with_retry(gd_id, self.files[0], self.output_dir, self.fn_mock_download)
            print("Downloaded", f"{self.output_dir}/{self.files[0]}")
        
        _check_cnt = 0
        while not os.path.exists(f"{self.output_dir}/{self.files[0]}") and _check_cnt<10:
            time.sleep(10)
            _check_cnt+=1
            
        if _check_cnt>=10:
            raise Exception(f"Unable to find file '{self.output_dir}/{self.files[0]}'")
            
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(f"{self.output_dir}/{self.files[self.current_shard]}")
        self.current_position = 0
        
        if len(self.files)>1 and self.local_process_rank==0:
            self.prepare_next_file()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(f"{self.output_dir}/{self.files[self.current_shard]}")
        
        if len(self.files)>1 and self.local_process_rank==0:
            self.prepare_next_file()
    
    def prepare_next_file(self):
        if len(self.files)==1:
            return
        
        next_shard = (self.current_shard + 1) % len(self.files)
        next_shard_file_name = self.files[next_shard]
        next_shard_file_name_gdid = self.file_names_dict[next_shard_file_name]
        next_shard_file = f"{self.output_dir}/{next_shard_file_name}"
        
        if not os.path.exists(next_shard_file) and self.local_process_rank==0:
            _download_file_in_background(next_shard_file_name_gdid, next_shard_file_name, self.output_dir, self.fn_mock_download)
                
    def next_batch(self):
        B = self.B
        T = self.T
        
        rank_position = self.current_position + self.process_rank * B * T
        buf = self.tokens[rank_position : rank_position+B*T+1]

        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y
    
    def replay_next_batch(self, it, grad_acc_steps=1):
        
        raise NotImplemented("NOT implemented!!!!")
            
        for i in range(it):
            for k in range(grad_acc_steps):
                self.next_batch()