import pathlib
import jax
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.shared.download as download
import numpy as np
import torch
from collections import OrderedDict
import orbax.checkpoint as ocp
from flax import traverse_util
import einops


def print_nested_dict(d, prefix=''):
    """递归打印嵌套字典的结构"""
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}/")
            print_nested_dict(value, prefix + '  ')
        else:
            if hasattr(value, 'shape'):
                print(f"{prefix}{key}: {value.shape}")
            else:
                print(f"{prefix}{key}: {type(value)}")

def download_and_load_structure():
    """下载并打印权重结构"""
    print("开始下载checkpoint...")
    checkpoint_path = "s3://openpi-assets/checkpoints/pi0_fast_base"
    local_path = download.maybe_download(checkpoint_path)
    print(f"成功下载checkpoint到: {local_path}")
    
    params_path = local_path / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")
    
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),
                    item
                ),
            ),
        )["params"]
    
    # 处理扁平化的参数字典
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    jax_params = traverse_util.unflatten_dict(flat_params)
    
    print("\n实际的JAX参数键:")
    print(list(jax_params.keys()))  # 打印顶层键
    
    # 如果params是嵌套的，可能需要进一步检查结构
    print("\n详细的参数结构:")
    print_nested_dict(jax_params)
    
    # 检查params是否在更深的层级
    if "params" in jax_params:
        jax_params = jax_params["params"]
    
    # 如果模型参数在特定的子键下
    if "model" in jax_params:
        jax_params = jax_params["model"]
        
    return jax_params

# 加载torch模型
def load_torch_params():
    torch_params = torch.load("/EFM-Pretrain/galaxea_0/runs/oxe_ema_pretrain--0215_090910/model_10000.pt", weights_only=True)
    torch_params = torch_params['model']
    
    # 打印PyTorch模型的详细结构和参数的shape信息
    
    # 去掉模型的前缀 
    new_torch_params = {}
    for key, value in torch_params.items():
        if key.startswith('_orig_mod.model.'):
            key = key[len('_orig_mod.model.'):]
        new_torch_params[key] = value
      
    # print("\nPyTorch模型结构和参数分布:")  
    # print_torch_model_structure(new_torch_params)
    
    return new_torch_params

def convert_array(x):
    """将JAX数组转换为PyTorch张量"""
    return torch.from_numpy(np.array(x))

def convert_linear(jax_w, jax_b=None):
    """转换线性层权重"""
    weight = convert_array(jax_w.T)
    if jax_b is not None:
        bias = convert_array(jax_b)
        if bias.size() != (weight.size(0),):
            raise ValueError(f"权重和偏置维度不匹配: 权重={weight.size()}, 偏置={bias.size()}")
    else:
        bias = None
    return weight, bias

def convert_jax_conv_to_torch(jax_weights, jax_bias=None):
    """
    将JAX/Flax的卷积层权重转换为PyTorch格式
    
    JAX权重格式: [height, width, in_channels, out_channels]
    PyTorch权重格式: [out_channels, in_channels, height, width]
    
    Args:
        jax_weights: JAX的卷积权重
        jax_bias: JAX的偏置项（可选）
    
    Returns:
        torch_weights: PyTorch格式的权重
        torch_bias: PyTorch格式的偏置（如果提供了jax_bias）
    """
    # 转换权重
    # 从[H, W, C_in, C_out]转到[C_out, C_in, H, W]
    torch_weights = jnp.transpose(jax_weights, (3, 2, 0, 1))
    torch_weights = torch.from_numpy(np.array(torch_weights))
    
    # 转换偏置（如果存在）
    torch_bias = None
    if jax_bias is not None:
        torch_bias = torch.from_numpy(np.array(jax_bias))
    
    return torch_weights, torch_bias


def convert_siglip(jax_params, torch_model):
    """转换Siglip视觉模型权重"""
    torch_state = {}
    # jax_params['PaliGemma']['img'].keys()
    # dict_keys(['Transformer', 'embedding', 'head', 'pos_embedding'])
    
    # 转换patch embedding
    w = jax_params['embedding']['kernel']
    b = jax_params['embedding']['bias']
    
    torch_state['vision_tower.vision_model.embeddings.patch_embedding.weight'], \
    torch_state['vision_tower.vision_model.embeddings.patch_embedding.bias'] = \
        convert_jax_conv_to_torch(w, b)
    
    # 转换position embedding
    pos_emb = jax_params['pos_embedding']
    torch_state['vision_tower.vision_model.embeddings.position_embedding.weight'] = \
        convert_array(pos_emb)[0] # [1, 256, 1152] -> [256, 1152]
    
    # vision_tower.vision_model.post_layernorm.weight
    w = jax_params['Transformer']['encoder_norm']['scale']
    b = jax_params['Transformer']['encoder_norm']['bias']
    torch_state['vision_tower.vision_model.post_layernorm.weight'] = convert_array(w)
    torch_state['vision_tower.vision_model.post_layernorm.bias'] = convert_array(b)
    
    layer_params = jax_params['Transformer']['encoderblock']
    # 转换Transformer层
    for i in range(27):
        prefix = f'vision_tower.vision_model.encoder.layers.{i}.'
        
        # jax_params['PaliGemma']['img']['Transformer'].keys()
        # dict_keys(['encoder_norm', 'encoderblock'])
        # encoderblock.keys()
        # ['LayerNorm_0', 'LayerNorm_1', 'MlpBlock_0', 'MultiHeadDotProductAttention_0']
        # MultiHeadDotProductAttention_0 : dict_keys(['key', 'out', 'query', 'value'])
        
        # 注意力层
        # 原始形状: [num_heads, seq_len, head_dim]
        q_w = layer_params['MultiHeadDotProductAttention_0']['query']['kernel'][i]  # [16, 72, 1152]
        q_b = layer_params['MultiHeadDotProductAttention_0']['query']['bias'][i]    # [16, 72]
        k_w = layer_params['MultiHeadDotProductAttention_0']['key']['kernel'][i]
        k_b = layer_params['MultiHeadDotProductAttention_0']['key']['bias'][i]
        v_w = layer_params['MultiHeadDotProductAttention_0']['value']['kernel'][i]
        v_b = layer_params['MultiHeadDotProductAttention_0']['value']['bias'][i]
        
        # 转换注意力层 flatten axis 0 and 1    
        q_w = q_w.reshape(q_w.shape[0], -1)
        k_w = k_w.reshape(k_w.shape[0], -1)
        v_w = v_w.reshape(v_w.shape[0], -1)
        q_b = q_b.reshape(-1)
        k_b = k_b.reshape(-1)
        v_b = v_b.reshape(-1)

        # 转换输出层
        # since the output is [16, 72, 1152], we need to flatten the axis 0 and 1
        out_w = layer_params['MultiHeadDotProductAttention_0']['out']['kernel'][i]
        out_b = layer_params['MultiHeadDotProductAttention_0']['out']['bias'][i] 
        out_w = out_w.reshape(-1, out_w.shape[-1])
        out_b = out_b.reshape(-1)
        
        torch_state[prefix + 'self_attn.q_proj.weight'], torch_state[prefix + 'self_attn.q_proj.bias'] = convert_linear(q_w, q_b)
        torch_state[prefix + 'self_attn.k_proj.weight'], torch_state[prefix + 'self_attn.k_proj.bias'] = convert_linear(k_w, k_b)
        torch_state[prefix + 'self_attn.v_proj.weight'], torch_state[prefix + 'self_attn.v_proj.bias'] = convert_linear(v_w, v_b)
        torch_state[prefix + 'self_attn.out_proj.weight'], torch_state[prefix + 'self_attn.out_proj.bias'] = convert_linear(out_w, out_b)
        
        # LayerNorm1
        layer_norm_w = layer_params['LayerNorm_0']['scale'][i]
        layer_norm_b = layer_params['LayerNorm_0']['bias'][i]
        torch_state[prefix + 'layer_norm1.weight'] = convert_array(layer_norm_w)
        torch_state[prefix + 'layer_norm1.bias'] = convert_array(layer_norm_b)
        
        # LayerNorm2
        layer_norm_w = layer_params['LayerNorm_1']['scale'][i]
        layer_norm_b = layer_params['LayerNorm_1']['bias'][i]
        torch_state[prefix + 'layer_norm2.weight'] = convert_array(layer_norm_w)
        torch_state[prefix + 'layer_norm2.bias'] = convert_array(layer_norm_b)   
        
        # MLP
        mlp_w1 = layer_params['MlpBlock_0']['Dense_0']['kernel'][i]
        mlp_b1 = layer_params['MlpBlock_0']['Dense_0']['bias'][i]
        torch_state[prefix + 'mlp.fc1.weight'], torch_state[prefix + 'mlp.fc1.bias'] = convert_linear(mlp_w1, mlp_b1)
        
        mlp_w2 = layer_params['MlpBlock_0']['Dense_1']['kernel'][i]
        mlp_b2 = layer_params['MlpBlock_0']['Dense_1']['bias'][i]
        torch_state[prefix + 'mlp.fc2.weight'], torch_state[prefix + 'mlp.fc2.bias'] = convert_linear(mlp_w2, mlp_b2)
         
    return torch_state

def convert_gemma_mixture(jax_params, torch_params):
    """转换Gemma混合模型权重"""
    torch_state = {}
    
    # 转换token embedding
    w = jax_params['embedder']['input_embedding']
    torch_state['embed_tokens.weight'] = torch_params['embed_tokens.weight']
    # TODO
    torch_state['embed_tokens.weight'][:257152] = convert_array(w)
    # hard code here since the input_embedding is 257152
    # the default vocab size is 257152 
    # ours tokenizer is 257216 FIXME: need to change the vocab size

    # 转换每个mixture层
    # VLM mixture (2048维)
    layer_params = jax_params['layers']
    # dict_keys(['attn', 'mlp', 'mlp_1', 'pre_attention_norm', 'pre_attention_norm_1', 'pre_ffw_norm', 'pre_ffw_norm_1'])
    for i in range(18):
        prefix = f'joint_model.mixtures.vlm.layers.{i}.'
        q_w = layer_params['attn']['q_einsum']['w'][i]  # [8, 2048, 256]
        q_w = einops.rearrange(q_w, 'H D O -> D (H O)')
        
        kv_w = layer_params['attn']['kv_einsum']['w'][i]  # [2, 1, 2048, 256]
        kv_w = einops.rearrange(kv_w, 'N H D O -> N D (H O)')
        
        o_w = layer_params['attn']['attn_vec_einsum']['w'][i]  # [8, 2048, 256]
        o_w = einops.rearrange(o_w, 'H D O -> (H D) O')
        
        # 转换注意力层
        torch_state[prefix + 'self_attn.q_proj.weight'], _ = convert_linear(q_w)  # 2048维权重
        torch_state[prefix + 'self_attn.k_proj.weight'], _ = convert_linear(kv_w[0])
        torch_state[prefix + 'self_attn.v_proj.weight'], _ = convert_linear(kv_w[1])
        torch_state[prefix + 'self_attn.o_proj.weight'], _ = convert_linear(o_w)
        
        # 转换RMSNorm
        torch_state[prefix + 'input_layernorm.weight'] = convert_array(layer_params['pre_attention_norm']['scale'][i])
        torch_state[prefix + 'post_attention_layernorm.weight'] = convert_array(layer_params['pre_ffw_norm']['scale'][i])
        
        # 转换MLP
        gate_proj_w = layer_params['mlp']['gating_einsum'][i]
        torch_state[prefix + 'mlp.gate_proj.weight'] = convert_array(gate_proj_w[0].T)
        torch_state[prefix + 'mlp.up_proj.weight'] = convert_array(gate_proj_w[1].T)
        torch_state[prefix + 'mlp.down_proj.weight'] = convert_array(layer_params['mlp']['linear'][i].T)

    torch_state['joint_model.mixtures.vlm.norm.weight'] = convert_array(jax_params['final_norm']['scale'])
    return torch_state

def convert_pi0fast_weights(jax_params, torch_params):
    """主转换函数"""
    torch_state = OrderedDict()
    
    # 1. 转换视觉模型 (Siglip)
    vision_state = convert_siglip(jax_params['PaliGemma']['img'], torch_params)
    torch_state.update(vision_state)
    
    # 2. 转换multi_modal_projector
    proj_w = jax_params['PaliGemma']['img']['head']['kernel']
    proj_b = jax_params['PaliGemma']['img']['head']['bias']
    torch_state['multi_modal_projector.linear.weight'], torch_state['multi_modal_projector.linear.bias'] = \
        convert_linear(proj_w, proj_b)
    
    # 3. 转换Joint Model (Mixtures)
    mixture_state = convert_gemma_mixture(jax_params['PaliGemma']['llm'], torch_params)
    torch_state.update(mixture_state)
    
    
    # check the torch_state and torch_params
    key_diff = set(torch_params.keys()) - set(torch_state.keys())
    if len(key_diff) > 0:
        print(f"torch_state has {len(key_diff)} more keys than torch_params")
        print(key_diff)
    
    # check the shape of the torch_state and torch_params
    for k, v in torch_state.items():
        if k in torch_params:
            if v.shape != torch_params[k].shape:
                print(f"ours {k}: {v.shape} and it should be {torch_params[k].shape}")
    
    import ipdb; ipdb.set_trace()
    torch_params.update(torch_state)
    # add model. prefix to the keys
    torch_params = {f'model.{k}': v for k, v in torch_params.items()}
        
    return torch_params

def load_pi0fast_weights(jax_params, torch_params):
    """加载并转换权重"""
    # 直接使用传入的jax_params
    torch_state = convert_pi0fast_weights(jax_params, torch_params)
    return torch_state

def print_torch_model_structure(params, prefix=''):
    """递归打印PyTorch模型的结构和参数数量"""
    for name, module in params.items():
        # 计算当前模块的shape
        shape = module.shape
        print(f"{prefix}{name}: {shape}")
       

if __name__ == "__main__":
    # 下载并打印权重结构
    # download.maybe_download(str("s3://openpi-assets/checkpoints/pi0_libero"))
    jax_params = download_and_load_structure()
    
    # 验证参数结构
    print("\n验证参数结构:")
    if not isinstance(jax_params, dict):
        raise TypeError(f"JAX参数应该是字典类型，但得到了 {type(jax_params)}")
    
    print("可用的顶层键:")
    print(list(jax_params.keys()))
    
    # 加载PyTorchc参数
    torch_params = load_torch_params()
    
    pi0fast_torch_state = load_pi0fast_weights(jax_params, torch_params)
    
    # save the torch_state
    torch.save({'model': pi0fast_torch_state}, "pi0fast_base.pt")