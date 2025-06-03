import time
import math
import argparse
import numpy as np
import pandas as pd

from numba import cuda, types # Added Numba CUDA imports

# Define a consistent data type for RIPEMD-160 words
DTYPE       = np.uint32
UINT32_TYPE = types.uint32 # Numba type for uint32

# File I/O and argparse (unchanged)
def writeFile(fname, code):
	try:
		with open(fname, 'wb') as f:
			f.write(''.join(code))
	except IOError:
		exit('No such file or directory ' + fname)

def readFile(fname):
	try:
		with open(fname, 'rb') as f:
			text = f.read()
	except IOError:
		exit('No such file or directory ' + fname)
	return text

def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('inFile')
	parser.add_argument('outFile')
	return parser.parse_args()

# CPU-side alignment function (unchanged, used by both CPU and GPU prep)
def alignment(msg_list):
	msg_len = len(msg_list) * 8
	msg_list.append(0x80)
	while len(msg_list) % 64 != 56:
		msg_list += [0]
	for i in range(8):
		msg_list.append(DTYPE((msg_len >> (i * 8)) & 0xFF))
	return msg_list

# --- GPU Device Functions ---
@cuda.jit(UINT32_TYPE(types.int32, UINT32_TYPE, UINT32_TYPE, UINT32_TYPE), device=True)
def F_device(j, x, y, z):
	if j < 16: return (x ^ y ^ z)
	if j < 32: return (x & y) | (~x & z)
	if j < 48: return (x | ~y) ^ z
	if j < 64: return (x & z) | (y & ~z)
	return x ^ (y | ~z)

@cuda.jit(UINT32_TYPE(types.int32), device=True)
def K_device(j):
	if j < 16: return DTYPE(0x00000000)
	if j < 32: return DTYPE(0x5A827999)
	if j < 48: return DTYPE(0x6ED9EBA1)
	if j < 64: return DTYPE(0x8F1BBCDC)
	return DTYPE(0xA953FD4E)

@cuda.jit(UINT32_TYPE(types.int32), device=True)
def K1_device(j):
	if j < 16: return DTYPE(0x50A28BE6)
	if j < 32: return DTYPE(0x5C4DD124)
	if j < 48: return DTYPE(0x6D703EF3)
	if j < 64: return DTYPE(0x7A6D76E9)
	return DTYPE(0x00000000)

@cuda.jit(UINT32_TYPE(UINT32_TYPE, types.int32), device=True)
def rotateLeft_device(x, n):
	return ((x << DTYPE(n)) | (x >> (DTYPE(32-n))))

R_arr_global = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
], dtype=np.int32)
R1_arr_global = np.array([
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
], dtype=np.int32)
S_arr_global = np.array([
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
], dtype=np.int32)
S1_arr_global = np.array([
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
], dtype=np.int32)

ROUNDS_SIG = types.void(
    UINT32_TYPE[:], UINT32_TYPE[:], 
    UINT32_TYPE[:], UINT32_TYPE[:], 
    types.int32[:], types.int32[:], types.int32[:], types.int32[:]
)
@cuda.jit(ROUNDS_SIG, device=True)
def rounds_device(buf_in, x_block, dataX_thread, dataY_thread, R_arr, R1_arr, S_arr, S1_arr):
	A=buf_in[0]; B=buf_in[1]; C=buf_in[2]; D=buf_in[3]; E=buf_in[4]
	A1=buf_in[0]; B1=buf_in[1]; C1=buf_in[2]; D1=buf_in[3]; E1=buf_in[4]
	for j in range(80):
		T_val = A + F_device(j,B,C,D) + x_block[R_arr[j]] + K_device(j)
		T_val = rotateLeft_device(T_val, S_arr[j]) + E
		A=E; E=D; D=rotateLeft_device(C,10); C=B; B=T_val
		dataX_thread[j] = B
		T_val1 = A1 + F_device(79-j,B1,C1,D1) + x_block[R1_arr[j]] + K1_device(j)
		T_val1 = rotateLeft_device(T_val1, S1_arr[j]) + E1
		A1=E1; E1=D1; D1=rotateLeft_device(C1,10); C1=B1; B1=T_val1
		dataY_thread[j] = B1
	T_comb=buf_in[1]+C+D1; buf_in[1]=buf_in[2]+D+E1; buf_in[2]=buf_in[3]+E+A1
	buf_in[3]=buf_in[4]+A+B1; buf_in[4]=buf_in[0]+B+C1; buf_in[0]=T_comb

KERNEL_SIG = types.void(
    UINT32_TYPE[:,:], UINT32_TYPE[:,:], UINT32_TYPE[:,:], UINT32_TYPE[:,:],
    types.int32[:], types.int32[:], types.int32[:], types.int32[:]
)
@cuda.jit(KERNEL_SIG)
def ripemd160_gpu_kernel(all_data_words_gpu, output_hashes_gpu, dataX_all_gpu, dataY_all_gpu, R_dev, R1_dev, S_dev, S1_dev):
	idx = cuda.grid(1)
	if idx < all_data_words_gpu.shape[0]:
		buf = cuda.local.array(5,dtype=DTYPE); dataX_thread = cuda.local.array(80,dtype=DTYPE); dataY_thread = cuda.local.array(80,dtype=DTYPE)
		buf[0]=DTYPE(0x67452301); buf[1]=DTYPE(0xefcdab89); buf[2]=DTYPE(0x98badcfe); buf[3]=DTYPE(0x10325476); buf[4]=DTYPE(0xc3d2e1f0)
		current_message_data_words = all_data_words_gpu[idx]
		# Assuming each message is one block (16 words) after padding
		x_block_slice = current_message_data_words[0:16] 
		rounds_device(buf, x_block_slice, dataX_thread, dataY_thread, R_dev, R1_dev, S_dev, S1_dev)
		for i in range(5): output_hashes_gpu[idx,i] = buf[i]
		for i in range(80): dataX_all_gpu[idx,i] = dataX_thread[i]; dataY_all_gpu[idx,i] = dataY_thread[i]

def toLittleEndian(word): # Used by both CPU and GPU output formatting (on CPU)
	word = DTYPE(word) 
	res=DTYPE(0); res|=((word>>DTYPE(0))&DTYPE(0xFF))<<DTYPE(24); res|=((word>>DTYPE(8))&DTYPE(0xFF))<<DTYPE(16)
	res|=((word>>DTYPE(16))&DTYPE(0xFF))<<DTYPE(8); res|=((word>>DTYPE(24))&DTYPE(0xFF))<<DTYPE(0)
	return res

WORDS_PER_MESSAGE_PADDED = 16 # Each message is 64 bytes / 16 words after padding

# Helper to prepare a single message's data_words (CPU side)
def prepare_single_message_data_words(message_idx_val):
    word1_hex = f"{message_idx_val:08X}"
    hex_data_string = f"{word1_hex}0000000000000000000000000000000000000000000000000000000000000000"
    byte_list_for_alignment = []
    for k_char in range(0, len(hex_data_string), 2):
        byte_list_for_alignment.append(int(hex_data_string[k_char:k_char+2], 16))
    padded_msg_bytes = alignment(byte_list_for_alignment)
    current_data_words = np.zeros(WORDS_PER_MESSAGE_PADDED, dtype=DTYPE)
    for word_idx in range(WORDS_PER_MESSAGE_PADDED):
        val = DTYPE(0)
        for byte_k in range(4):
            val |= DTYPE(padded_msg_bytes[word_idx * 4 + byte_k]) << DTYPE(byte_k * 8)
        current_data_words[word_idx] = val
    return current_data_words, hex_data_string # Return hex_data_string for debug prints

if __name__ == '__main__':
	script_start_time = time.time()

	# Ensure device constants are available for single GPU test runs
	d_R_arr_test = cuda.to_device(R_arr_global)
	d_R1_arr_test = cuda.to_device(R1_arr_global)
	d_S_arr_test = cuda.to_device(S_arr_global)
	d_S1_arr_test = cuda.to_device(S1_arr_global)
	
	# --- Main GPU Batch Processing ---
	print("\n--- Starting Main GPU Batch Processing ---")
	rowList    = []
	total      = 0x100000
	range_stop = total
	
	cpu_prep_start_time = time.time()
	# ... (rest of the main script for batch processing, timings, CSV saving)
	# This part is identical to the previous version of the script.
	# For brevity, I'm omitting the copy-paste of the batch processing part.
	# Assume it starts here.
	print("Preparing input data on CPU for batch...")
	all_message_data_words_list = []
	for i in range(range_stop):
		# Using the helper for consistency, though it recalculates hex_data_string
		current_data_words, _ = prepare_single_message_data_words(i)
		all_message_data_words_list.append(current_data_words)

	all_data_words_np    = np.stack(all_message_data_words_list, axis=0)
	cpu_prep_end_time    = time.time()
	cpu_preparation_time = cpu_prep_end_time - cpu_prep_start_time
	print(f"CPU Data Preparation time for batch: {cpu_preparation_time:.4f} seconds")
	print(f"Input data shape for batch: {all_data_words_np.shape}")
	
	start_gpu_total_time = time.time()
	print("Transferring batch data to GPU and allocating GPU memory...")
	d_all_data_words = cuda.to_device(all_data_words_np)
	d_output_hashes  = cuda.device_array((total, 5), dtype=DTYPE) 
	d_dataX_all      = cuda.device_array((total, 80), dtype=DTYPE)
	d_dataY_all      = cuda.device_array((total, 80), dtype=DTYPE)

	# Constants are already on device from test, or re-copy (d_R_arr_test etc. can be reused if scope allows,
	# but for clarity, let's assume they might be separate or re-copied if this was a standalone function)
	# Re-using d_R_arr_test, d_R1_arr_test, d_S_arr_test, d_S1_arr_test from the test section for the main batch.
	# If test section were conditional, these would need to be defined here too.
	
	threadsperblock = 128 
	blockspergrid   = (total + (threadsperblock - 1)) // threadsperblock

	start_kernel_execution_time = time.time()
	print(f"Launching CUDA kernel for batch with {blockspergrid} blocks and {threadsperblock} threads per block...")
	ripemd160_gpu_kernel[blockspergrid, threadsperblock](
		d_all_data_words, d_output_hashes, d_dataX_all, d_dataY_all,
		d_R_arr_test, d_R1_arr_test, d_S_arr_test, d_S1_arr_test # Re-use device constants
	)
	cuda.synchronize() 
	end_kernel_execution_time = time.time()
	print("Batch Kernel execution finished.")

	start_copy_back_time = time.time()
	print("Copying batch results back to CPU...")
	output_hashes_cpu  = d_output_hashes.copy_to_host()
	dataX_batch_cpu    = d_dataX_all.copy_to_host() # Renamed to avoid conflict with test variable
	dataY_batch_cpu    = d_dataY_all.copy_to_host() # Renamed
	end_copy_back_time = time.time()
	end_gpu_total_time = time.time()

	time_to_kernel_launch = start_kernel_execution_time - start_gpu_total_time
	kernel_execution_time = end_kernel_execution_time - start_kernel_execution_time
	data_transfer_from_gpu_time = end_copy_back_time - start_copy_back_time
	total_gpu_processing_time   = end_gpu_total_time - start_gpu_total_time
	
	print("\n--- Performance Metrics for Batch Processing ---")
	print(f"Time for CPU data preparation: {cpu_preparation_time:.4f} seconds") # This is batch prep
	print(f"Time for GPU allocations and data transfer to GPU: {time_to_kernel_launch:.4f} seconds")
	print(f"Kernel execution time: {kernel_execution_time:.4f} seconds")
	print(f"Data transfer back from GPU time: {data_transfer_from_gpu_time:.4f} seconds")
	print(f"Total GPU processing time (including transfers): {total_gpu_processing_time:.4f} seconds")

	cpu_post_processing_start_time = time.time()
	print("\nProcessing batch results on CPU and saving to CSV...")
	for i in range(total):
		hash_hex_str = ""
		for k_hash_word in range(5):
			hash_hex_str += "{:08X} ".format(toLittleEndian(output_hashes_cpu[i, k_hash_word]))
		
		dict1 = {
			'message_words': all_data_words_np[i].tolist(), 
			'hash_output': hash_hex_str.strip(),
			'X_intermediates': dataX_batch_cpu[i].tolist(), # Use batch variables
			'Y_intermediates': dataY_batch_cpu[i].tolist()  # Use batch variables
		}
		rowList.append(dict1)

		if i % 10000 == 0 or i == total - 1: 
			percent = (i + 1) * 100 / total
			print(f"CPU Post-processing & CSV creation for batch: {percent:.2f}% completed.")

		if i % 0x1FFFFF == 0 and i > 0: 
			df = pd.DataFrame(rowList); df.to_csv(f'm0x0-gpu-part-{i//0x1FFFFF:03d}.csv', index=False); rowList = []
	if rowList: 
		df = pd.DataFrame(rowList); df.to_csv(f'm0x0-gpu-part-final.csv', index=False)
	
	cpu_post_processing_end_time = time.time()
	cpu_post_processing_time = cpu_post_processing_end_time - cpu_post_processing_start_time
	print(f"CPU Post-processing and CSV saving time for batch: {cpu_post_processing_time:.4f} seconds")

	script_end_time = time.time()
	total_script_time = script_end_time - script_start_time
	print(f"\nTotal script execution time (including tests): {total_script_time:.4f} seconds")
	print("\nDone processing and saving data with GPU.")
