from datasets import load_dataset
import re

# Load the dataset
dataset = load_dataset("GPUMODE/Inductor_Created_Data_Permissive", split="train")
print(f"Total examples: {len(dataset)}")

# Function to extract get_init_inputs function
def extract_get_init_inputs(code):
    # Regex pattern to find get_init_inputs function
    pattern = r'def get_init_inputs\(\):.*?return.*?(?=\n\n|\n\w|$)'
    matches = re.findall(pattern, code, re.DOTALL)
    if matches:
        return matches[0]
    return None

# Sample 10 examples
print("\nSampling get_init_inputs from 10 examples:")
for i in range(10):
    code = dataset[i]["python_code"]
    function = extract_get_init_inputs(code)
    if function:
        print(f"\n=== EXAMPLE {i+1} ===")
        print(function) 