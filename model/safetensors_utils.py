import os
import torch
import math
from typing import Dict, Any, List
from safetensors import safe_open
from safetensors.torch import save_file

def calculate_chunk_size(tensor_size: int, target_size_mb: float = 2.4) -> int:
    target_size_bytes = target_size_mb * 1024 * 1024
    element_size = 4
    return target_size_bytes // element_size

def split_tensor_into_chunks(tensor: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    total_elements = tensor.numel()
    chunks = []
    
    for i in range(0, total_elements, chunk_size):
        end_idx = min(i + chunk_size, total_elements)
        chunk = tensor.view(-1)[i:end_idx]
        chunks.append(chunk)
    
    return chunks

def save_model_safetensors(model: torch.nn.Module, base_path: str, target_size_mb: float = 2.4):
    print(f"Saving Model as Safetensors (max {target_size_mb}MB per file)")
    
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    state_dict = model.state_dict()
    chunk_size = calculate_chunk_size(1, target_size_mb)
    
    current_chunk = {}
    current_size = 0
    chunk_index = 0
    
    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * 4
        
        if current_size + tensor_size > target_size_mb * 1024 * 1024 and current_chunk:
            chunk_path = f"{base_path}_chunk_{chunk_index:03d}.safetensors"
            save_file(current_chunk, chunk_path)
            print(f"Saved chunk {chunk_index}: {chunk_path} ({current_size / (1024*1024):.2f}MB)")
            
            current_chunk = {}
            current_size = 0
            chunk_index += 1
        
        if tensor_size > target_size_mb * 1024 * 1024:
            tensor_chunks = split_tensor_into_chunks(tensor, chunk_size)
            for i, chunk in enumerate(tensor_chunks):
                chunk_name = f"{name}_chunk_{i}"
                current_chunk[chunk_name] = chunk
                current_size += chunk.numel() * 4
                
                if current_size >= target_size_mb * 1024 * 1024:
                    chunk_path = f"{base_path}_chunk_{chunk_index:03d}.safetensors"
                    save_file(current_chunk, chunk_path)
                    print(f"Saved chunk {chunk_index}: {chunk_path} ({current_size / (1024*1024):.2f}MB)")
                    
                    current_chunk = {}
                    current_size = 0
                    chunk_index += 1
        else:
            current_chunk[name] = tensor
            current_size += tensor_size
    
    if current_chunk:
        chunk_path = f"{base_path}_chunk_{chunk_index:03d}.safetensors"
        save_file(current_chunk, chunk_path)
        print(f"Saved chunk {chunk_index}: {chunk_path} ({current_size / (1024*1024):.2f}MB)")
    
    metadata_path = f"{base_path}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump({
            'total_chunks': chunk_index + 1,
            'target_size_mb': target_size_mb,
            'model_config': model.config.__dict__ if hasattr(model, 'config') else {}
        }, f, indent=2)
    
    print(f"Model saved in {chunk_index + 1} chunks")

def load_model_safetensors(model_class, base_path: str, device: str = 'cpu'):
    print(f"Loading Model from Safetensors")
    
    metadata_path = f"{base_path}_metadata.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_chunks = metadata['total_chunks']
    state_dict = {}
    
    for chunk_idx in range(total_chunks):
        chunk_path = f"{base_path}_chunk_{chunk_idx:03d}.safetensors"
        if not os.path.exists(chunk_path):
            print(f"Warning: Chunk {chunk_idx} not found: {chunk_path}")
            continue
        
        print(f"Loading chunk {chunk_idx}: {chunk_path}")
        
        with safe_open(chunk_path, framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                if '_chunk_' in key:
                    base_name = key.split('_chunk_')[0]
                    if base_name not in state_dict:
                        state_dict[base_name] = []
                    state_dict[base_name].append(tensor)
                else:
                    state_dict[key] = tensor
    
    for key, value in state_dict.items():
        if isinstance(value, list):
            state_dict[key] = torch.cat(value, dim=0)
    
    if 'model_config' in metadata and metadata['model_config']:
        from main.config import ModelConfig
        config = ModelConfig(**metadata['model_config'])
        model = model_class(config)
    else:
        model = model_class()
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"Model loaded successfully from {total_chunks} chunks")
    
    return model

def get_model_size_info(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    param_info = {}
    for name, param in model.named_parameters():
        param_info[name] = {
            'shape': list(param.shape),
            'numel': param.numel(),
            'size_mb': param.numel() * param.element_size() / (1024 * 1024)
        }
    
    return {
        'total_parameters': total_params,
        'total_size_mb': total_size / (1024 * 1024),
        'parameters': param_info
    }
