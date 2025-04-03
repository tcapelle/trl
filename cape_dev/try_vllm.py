#!/usr/bin/env python
# try_vllm.py - Simple script to demonstrate vLLM with Qwen/Qwen2.5-Coder-3B-Instruct

import time
import torch
from vllm import LLM, SamplingParams

def main():
    # Print basic info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Load the model
    print("\nLoading model 'Qwen/Qwen2.5-Coder-3B-Instruct'...")
    start_time = time.time()
    
    # Initialize the LLM with vLLM
    # Setting a reasonable gpu_memory_utilization to avoid OOM errors
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        gpu_memory_utilization=0.85,  # Adjust as needed based on your GPU
        dtype="float16",  # Using float16 to reduce memory requirements
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Define prompts for inference
    prompts = [
        "Write a Python function to calculate the Fibonacci sequence up to n terms.",
        "Explain how to use async/await in JavaScript with a simple example."
    ]
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    
    # Run inference
    print("\nRunning inference on sample prompts...")
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - start_time
    
    # Process and display results
    print(f"\nInference completed in {inference_time:.2f} seconds")
    print(f"Generated {len(outputs)} responses")
    
    # Print the generated text for each prompt
    for i, output in enumerate(outputs):
        print(f"\n\nPrompt {i+1}: {prompts[i]}")
        print(f"\nGenerated output:")
        print(f"{output.outputs[0].text}")
        print("=" * 80)
    
    # Print model information
    max_model_len = llm.get_max_context_length()
    print(f"\nModel max context length: {max_model_len}")

if __name__ == "__main__":
    main()
