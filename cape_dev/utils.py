import hashlib
import re
# ===== Utility Functions =====

def extract_python_code(text):
    """Extract Python code from markdown code blocks or text"""
    # Look for Python code blocks
    python_blocks = re.findall(r'```(?:python|py)?\s*([\s\S]*?)```', text)
    if python_blocks:
        return python_blocks[0].strip()
        
    # If no Python blocks, extract anything that looks like Python code
    code_lines = []
    in_code = False
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else text


# Helper function to ensure content is a string
def ensure_string(content):
    """Convert content to string if it's not already"""
    if isinstance(content, list):
        return ensure_string(content[0])
    elif not isinstance(content, str):
        return str(content)
    return content

# Helper function to create a hash of content for caching
def hash_content(content):
    """Create a hash of content for use as a cache key"""
    content_str = ensure_string(content)
    return hashlib.md5(content_str.encode()).hexdigest()


# Custom JSON serializer to handle non-serializable objects
def serialize_for_wandb(obj):
    """Convert training config to a wandb-compatible dict, handling non-serializable objects."""
    if hasattr(obj, "__dict__"):
        return {k: serialize_for_wandb(v) for k, v in obj.__dict__.items() 
                if not k.startswith("_")}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_wandb(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_wandb(v) for k, v in obj.items()}
    elif hasattr(obj, "to_dict"):
        try:
            return serialize_for_wandb(obj.to_dict())
        except:
            return str(obj)
    else:
        # For objects that can't be serialized, convert to string
        try:
            return obj
        except:
            return str(obj)
