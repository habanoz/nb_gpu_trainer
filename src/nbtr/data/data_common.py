"""
Modification of https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py
###############
Common utilities for the datasets
"""
import numpy as np

HEADERS_INFO = {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    }

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    info = HEADERS_INFO
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes)")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())