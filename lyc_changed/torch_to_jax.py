import jax.numpy as jnp
import torch
import einops
from flax import traverse_util
from etils import epath
import os


def convert_array(x):
    """将PyTorch张量转换为JAX数组"""
    return jnp.array(x.detach().cpu().numpy())

def convert_linear(torch_w, torch_b=None):
    """转换线性层权重从PyTorch到JAX格式"""
    weight = convert_array(torch_w.T)  # PyTorch到JAX需要转置
    if torch_b is not None:
        bias = convert_array(torch_b)
    else:
        bias = None
    return weight, bias

def convert_torch_conv_to_jax(torch_weights, torch_bias=None):
    """
    将PyTorch的卷积层权重转换为JAX/Flax格式
    
    PyTorch权重格式: [out_channels, in_channels, height, width]
    JAX权重格式: [height, width, in_channels, out_channels]
    """
    # 转换权重
    # 从[C_out, C_in, H, W]转到[H, W, C_in, C_out]
    jax_weights = jnp.transpose(convert_array(torch_weights), (2, 3, 1, 0))
    
    # 转换偏置（如果存在）
    jax_bias = None
    if torch_bias is not None:
        jax_bias = convert_array(torch_bias)
    
    return jax_weights, jax_bias

def convert_siglip_to_jax(torch_state):
    """将Siglip视觉模型的PyTorch权重转换为JAX格式"""
    jax_params = {
        'embedding': {},
        'pos_embedding': None,
        'Transformer': {
            'encoder_norm': {},
            'encoderblock': {
                'LayerNorm_0': {'scale': [], 'bias': []},
                'LayerNorm_1': {'scale': [], 'bias': []},
                'MultiHeadDotProductAttention_0': {
                    'query': {'kernel': [], 'bias': []},
                    'key': {'kernel': [], 'bias': []},
                    'value': {'kernel': [], 'bias': []},
                    'out': {'kernel': [], 'bias': []}
                },
                'MlpBlock_0': {
                    'Dense_0': {'kernel': [], 'bias': []},
                    'Dense_1': {'kernel': [], 'bias': []}
                }
            }
        }
    }
    
    # 转换patch embedding
    w, b = convert_torch_conv_to_jax(
        torch_state['vision_tower.vision_model.embeddings.patch_embedding.weight'],
        torch_state['vision_tower.vision_model.embeddings.patch_embedding.bias']
    )
    jax_params['embedding']['kernel'] = w
    jax_params['embedding']['bias'] = b
    
    # 转换position embedding
    pos_emb = convert_array(torch_state['vision_tower.vision_model.embeddings.position_embedding.weight'])
    jax_params['pos_embedding'] = jnp.expand_dims(pos_emb, 0)  # [256, 1152] -> [1, 256, 1152]
    
    # 转换encoder norm
    jax_params['Transformer']['encoder_norm']['scale'] = convert_array(
        torch_state['vision_tower.vision_model.post_layernorm.weight']
    )
    jax_params['Transformer']['encoder_norm']['bias'] = convert_array(
        torch_state['vision_tower.vision_model.post_layernorm.bias']
    )
    
    # 转换Transformer层
    for i in range(27):
        prefix = f'vision_tower.vision_model.encoder.layers.{i}.'
        
        # 注意力层
        q_w, q_b = convert_linear(torch_state[prefix + 'self_attn.q_proj.weight'],
                                  torch_state[prefix + 'self_attn.q_proj.bias'])
        k_w, k_b = convert_linear(torch_state[prefix + 'self_attn.k_proj.weight'],
                                  torch_state[prefix + 'self_attn.k_proj.bias'])
        v_w, v_b = convert_linear(torch_state[prefix + 'self_attn.v_proj.weight'],
                                  torch_state[prefix + 'self_attn.v_proj.bias'])
        
        # 重塑为JAX格式
        q_w = einops.rearrange(q_w, 'D (H O) -> D H O', H=16)
        k_w = einops.rearrange(k_w, 'D (H O) -> D H O', H=16)
        v_w = einops.rearrange(v_w, 'D (H O) -> D H O', H=16)
        q_b = einops.rearrange(q_b, '(H O) -> H O', H=16)
        k_b = einops.rearrange(k_b, '(H O) -> H O', H=16)
        v_b = einops.rearrange(v_b, '(H O) -> H O', H=16)
        
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['query']['kernel'].append(q_w)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['query']['bias'].append(q_b)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['kernel'].append(k_w)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['bias'].append(k_b)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['value']['kernel'].append(v_w)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['value']['bias'].append(v_b)
        
        # 输出层
        out_w, out_b = convert_linear(torch_state[prefix + 'self_attn.out_proj.weight'],
                                      torch_state[prefix + 'self_attn.out_proj.bias'])
        out_w = einops.rearrange(out_w, '(H O) D -> H O D', H=16)
        
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['out']['kernel'].append(out_w)
        jax_params['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['out']['bias'].append(out_b)
        
        # LayerNorms
        jax_params['Transformer']['encoderblock']['LayerNorm_0']['scale'].append(
            convert_array(torch_state[prefix + 'layer_norm1.weight'])
        )
        jax_params['Transformer']['encoderblock']['LayerNorm_0']['bias'].append(
            convert_array(torch_state[prefix + 'layer_norm1.bias'])
        )
        jax_params['Transformer']['encoderblock']['LayerNorm_1']['scale'].append(
            convert_array(torch_state[prefix + 'layer_norm2.weight'])
        )
        jax_params['Transformer']['encoderblock']['LayerNorm_1']['bias'].append(
            convert_array(torch_state[prefix + 'layer_norm2.bias'])
        )
        
        # MLP
        jax_params['Transformer']['encoderblock']['MlpBlock_0']['Dense_0']['kernel'].append(
            convert_array(torch_state[prefix + 'mlp.fc1.weight'].T)
        )
        jax_params['Transformer']['encoderblock']['MlpBlock_0']['Dense_0']['bias'].append(
            convert_array(torch_state[prefix + 'mlp.fc1.bias'])
        )
        jax_params['Transformer']['encoderblock']['MlpBlock_0']['Dense_1']['kernel'].append(
            convert_array(torch_state[prefix + 'mlp.fc2.weight'].T)
        )
        jax_params['Transformer']['encoderblock']['MlpBlock_0']['Dense_1']['bias'].append(
            convert_array(torch_state[prefix + 'mlp.fc2.bias'])
        )
    
    # 将列表转换为数组
    for key_path, value in traverse_util.flatten_dict(jax_params).items():
        if isinstance(value, list):
            parent = jax_params
            for k in key_path[:-1]:
                parent = parent[k]
            parent[key_path[-1]] = jnp.stack(value)
    
    return jax_params

def convert_gemma_mixture_to_jax(torch_state):
    """将Gemma混合模型的PyTorch权重转换为JAX格式"""
    jax_params = {
        'embedder': {'input_embedding': None},
        'layers': {
            'attn': {
                'q_einsum': {'w': []},
                'kv_einsum': {'w': []},
                'attn_vec_einsum': {'w': []},
                'q_einsum_1': {'w': []},
                'kv_einsum_1': {'w': []},
                'attn_vec_einsum_1': {'w': []}
            },
            'pre_attention_norm': {'scale': []},
            'pre_attention_norm_1': {'scale': []},
            'pre_ffw_norm': {'scale': []},
            'pre_ffw_norm_1': {'scale': []},
            'mlp': {
                'gating_einsum': [],
                'linear': []
            },
            'mlp_1': {
                'gating_einsum': [],
                'linear': []
            }
        },
        'final_norm_1': {'scale': None}    
    }
    
    # 转换token embedding
    jax_params['embedder']['input_embedding'] = convert_array(
        torch_state['embed_tokens.weight'][:257152]  # 只取前257152个token
    )
    
    # 转换VLM mixture层 (2048维)
    for i in range(18):
        prefix = f'joint_model.mixtures.vlm.layers.{i}.'
        
        # 注意力层
        q_w, _ = convert_linear(torch_state[prefix + 'self_attn.q_proj.weight'])
        q_w = einops.rearrange(q_w, 'D (H O) -> H D O', H=8)
        
        k_w, _ = convert_linear(torch_state[prefix + 'self_attn.k_proj.weight'])
        v_w, _ = convert_linear(torch_state[prefix + 'self_attn.v_proj.weight'])
        kv_w = jnp.stack([k_w, v_w])
        kv_w = einops.rearrange(kv_w, 'N D (H O) -> N H D O', H=1)
        
        o_w, _ = convert_linear(torch_state[prefix + 'self_attn.o_proj.weight'])
        o_w = einops.rearrange(o_w, '(H D) O -> H D O', H=8)
        
        jax_params['layers']['attn']['q_einsum']['w'].append(q_w)
        jax_params['layers']['attn']['kv_einsum']['w'].append(kv_w)
        jax_params['layers']['attn']['attn_vec_einsum']['w'].append(o_w)
        
        # RMSNorm
        jax_params['layers']['pre_attention_norm']['scale'].append(
            convert_array(torch_state[prefix + 'input_layernorm.weight'])
        )
        jax_params['layers']['pre_ffw_norm']['scale'].append(
            convert_array(torch_state[prefix + 'post_attention_layernorm.weight'])
        )
        
        # MLP
        gate_w, _ = convert_linear(torch_state[prefix + 'mlp.gate_proj.weight'])
        up_w, _ = convert_linear(torch_state[prefix + 'mlp.up_proj.weight'])
        jax_params['layers']['mlp']['gating_einsum'].append(jnp.stack([gate_w, up_w]))
        jax_params['layers']['mlp']['linear'].append(
            convert_linear(torch_state[prefix + 'mlp.down_proj.weight'])[0]
        )
    
    # Final norm
    # import ipdb; ipdb.set_trace()
    # jax_params['final_norm']['scale'] = convert_array(
    #     torch_state['joint_model.mixtures.vlm.norm.weight']
    # )
    
    # 转换Action/Proprio mixture层 (1024维)
    for i in range(18):
        prefix = f'joint_model.mixtures.action.layers.{i}.'
        
        # 注意力层
        q_w, _ = convert_linear(torch_state[prefix + 'self_attn.q_proj.weight'])
        q_w = einops.rearrange(q_w, 'D (H O) -> H D O', H=8)
                
        k_w, _ = convert_linear(torch_state[prefix + 'self_attn.k_proj.weight'])
        v_w, _ = convert_linear(torch_state[prefix + 'self_attn.v_proj.weight'])
        kv_w = jnp.stack([k_w, v_w])
        kv_w = einops.rearrange(kv_w, 'N D (H O) -> N H D O', H=1)
        
        o_w, _ = convert_linear(torch_state[prefix + 'self_attn.o_proj.weight'])
        o_w = einops.rearrange(o_w, '(H D) O -> H D O', H=8)
        
        jax_params['layers']['attn']['q_einsum_1']['w'].append(q_w)
        jax_params['layers']['attn']['kv_einsum_1']['w'].append(kv_w)
        jax_params['layers']['attn']['attn_vec_einsum_1']['w'].append(o_w)
        
        # RMSNorm
        jax_params['layers']['pre_attention_norm_1']['scale'].append(
            convert_array(torch_state[prefix + 'input_layernorm.weight'])
        )
        jax_params['layers']['pre_ffw_norm_1']['scale'].append(
            convert_array(torch_state[prefix + 'post_attention_layernorm.weight'])
        )
        
        # MLP
        gate_w, _ = convert_linear(torch_state[prefix + 'mlp.gate_proj.weight'])
        up_w, _ = convert_linear(torch_state[prefix + 'mlp.up_proj.weight'])
        jax_params['layers']['mlp_1']['gating_einsum'].append(jnp.stack([gate_w, up_w]))
        jax_params['layers']['mlp_1']['linear'].append(
            convert_linear(torch_state[prefix + 'mlp.down_proj.weight'])[0]
        )
    
    # Final norm
    jax_params['final_norm_1']['scale'] = convert_array(
        torch_state['joint_model.mixtures.action.norm.weight']
    )
    
    # 将列表转换为数组
    for key_path, value in traverse_util.flatten_dict(jax_params).items():
        if isinstance(value, list):
            parent = jax_params
            for k in key_path[:-1]:
                parent = parent[k]
            parent[key_path[-1]] = jnp.stack(value)
    
    return jax_params

def convert_torch_to_jax(torch_state):
    """主转换函数"""
    jax_params = {
        'PaliGemma': {
            'img': None,  # Siglip
            'llm': None,  # Gemma
        },
        'state_proj': {},
        'action_in_proj': {},
        'action_time_mlp_in': {},
        'action_time_mlp_out': {},
        'action_out_proj': {}
    }
    
    # 1. 转换视觉模型 (Siglip)
    vision_params = {}
    for k in torch_state.keys():
        if k.startswith('vision_tower.'):
            vision_params[k] = torch_state[k]
    jax_params['PaliGemma']['img'] = convert_siglip_to_jax(vision_params)
    
    # 2. 转换multi_modal_projector
    w, b = convert_linear(
        torch_state['multi_modal_projector.linear.weight'],
        torch_state['multi_modal_projector.linear.bias']
    )
    jax_params['PaliGemma']['img']['head'] = {
        'kernel': w,
        'bias': b
    }
    
    # 3. 转换Joint Model (Mixtures)
    mixture_params = {}
    for k in torch_state.keys():
        if k.startswith('joint_model.'):
            mixture_params[k] = torch_state[k]
    mixture_params['embed_tokens.weight'] = torch_state['embed_tokens.weight']
    jax_params['PaliGemma']['llm'] = convert_gemma_mixture_to_jax(mixture_params)
    
    # 4. 转换其他组件（如果存在）
    if 'proprio_encoder.weight' in torch_state:
        w, b = convert_linear(
            torch_state['proprio_encoder.weight'],
            torch_state['proprio_encoder.bias']
        )
        jax_params['state_proj']['kernel'] = w
        jax_params['state_proj']['bias'] = b
        
        # Action encoders
        w, b = convert_linear(
            torch_state['action_encoder.linear_1.weight'],
            torch_state['action_encoder.linear_1.bias']
        )
        jax_params['action_in_proj']['kernel'] = w
        jax_params['action_in_proj']['bias'] = b
        
        w, b = convert_linear(
            torch_state['action_encoder.linear_2.weight'],
            torch_state['action_encoder.linear_2.bias']
        )
        jax_params['action_time_mlp_in']['kernel'] = w
        jax_params['action_time_mlp_in']['bias'] = b
        
        w, b = convert_linear(
            torch_state['action_encoder.linear_3.weight'],
            torch_state['action_encoder.linear_3.bias']
        )
        jax_params['action_time_mlp_out']['kernel'] = w
        jax_params['action_time_mlp_out']['bias'] = b
        
        # Action decoder
        w, b = convert_linear(
            torch_state['action_decoder.weight'],
            torch_state['action_decoder.bias']
        )
        jax_params['action_out_proj']['kernel'] = w
        jax_params['action_out_proj']['bias'] = b
    
    return jax_params

def save_jax_params(jax_params, save_path):
    """保存JAX参数到文件"""
    # 创建保存目录
    save_path = os.path.abspath(save_path)
    save_path = epath.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存JAX参数 成一个pt文件，使用 pickle_protocol=5
    torch.save(
        jax_params, 
        save_path / "jax_params.pt",
        pickle_protocol=5  # 使用更高版本的pickle协议
    )
   
    print(f"成功保存JAX参数到: {save_path}")

def compare_params(params1, params2, prefix=""):
    """比较两个参数字典的结构和值"""
    # 展平两个字典
    flat1 = traverse_util.flatten_dict(params1)
    flat2 = traverse_util.flatten_dict(params2)
    
    # 比较键集
    keys1 = set(flat1.keys())
    keys2 = set(flat2.keys())
    
    # 找出不同的键
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2
    
    print("\n=== 参数结构比较 ===")
    if only_in_1:
        print("\n仅在params1中存在的键:")
        for k in sorted(only_in_1):
            print(f"  {'.'.join(str(x) for x in k)}")
    
    if only_in_2:
        print("\n仅在params2中存在的键:")
        for k in sorted(only_in_2):
            print(f"  {'.'.join(str(x) for x in k)}")
    
    print("\n=== 参数值比较 ===")
    # 比较共同键的值
    for k in sorted(common_keys):
        v1 = flat1[k]
        v2 = flat2[k]
        
        # 检查形状
        if hasattr(v1, 'shape') and hasattr(v2, 'shape'):
            if v1.shape != v2.shape:
                print(f"\n形状不同 - {'.'.join(str(x) for x in k)}:")
                print(f"  params1: {v1.shape}")
                print(f"  params2: {v2.shape}")
                continue
        
        # 检查值
        try:
            if isinstance(v1, jnp.ndarray) and isinstance(v2, jnp.ndarray):
                diff = jnp.abs(v1 - v2).mean()
                max_diff = jnp.abs(v1 - v2).max()
                if diff > 1e-5:  # 设置一个阈值
                    print(f"\n值不同 - {'.'.join(str(x) for x in k)}:")
                    print(f"  平均差异: {diff:.6f}")
                    print(f"  最大差异: {max_diff:.6f}")
        except Exception as e:
            print(f"\n比较出错 - {'.'.join(str(x) for x in k)}:")
            print(f"  错误: {str(e)}")

if __name__ == "__main__":
    # 加载PyTorch模型权重
    torch_state = torch.load("/EFM-Pretrain/galaxea_0/runs/ft_libero_pi_wrist_cam_fp32_halfdata--pi_half_data/model_15000.pt")
    if 'model' in torch_state:
        torch_state = torch_state['model']
    
    # 移除开头的'model.'前缀
    new_state_dict = {}
    for k, v in torch_state.items():
        if "_orig_mod" in k:
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    torch_state = {k[6:] if k.startswith('model.') else k: v for k, v in new_state_dict.items()}
    
    # 转换为JAX格式
    jax_params = convert_torch_to_jax(torch_state)
    
    # 对比load 原始的jax参数
    import sys
    sys.path.append("/EFM-Pretrain/lyc/openpi/")
    from lyc_changed.load_model import download_and_load_structure
    jax_params_origin = download_and_load_structure()
    
    # 比较参数
    print("\n开始比较参数...")
    compare_params(jax_params, jax_params_origin)
    
    # 保存JAX参数
    save_jax_params(jax_params, "jax_params") 