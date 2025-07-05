import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch
import scipy.special
import os

# Initialize distilroberta-base
try:
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModel.from_pretrained("distilroberta-base", use_safetensors=True)
except Exception as e:
    print(f"Error loading distilroberta-base: {e}")
    print("Ensure model is downloaded or check network connection.")
    raise

# Pre-fit PCA on mock SmartBugs snippets
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

mock_snippets = [
    {'jumpSeq': 'PUSH1 0x40 JUMPI', 'pc': '0x12'},
    {'jumpSeq': 'PUSH1 0x10 JUMP', 'pc': '0x1a'},
    {'jumpSeq': 'PUSH2 0x100 JUMPI', 'pc': '0x20'},
    {'jumpSeq': 'DUP1 SWAP1 JUMP', 'pc': '0x30'}
]
try:
    embeddings_list = [get_embeddings(f"{s['jumpSeq']} | pc:{s['pc']}") for s in mock_snippets]
    pca = PCA(n_components=3)
    pca.fit(embeddings_list)
except Exception as e:
    print(f"Error pre-fitting PCA: {e}")
    raise

# Projection matrices with increased variance for SSI
np.random.seed(42)
d_intermediate = 8
W_q = np.random.normal(0, 2.0, (3, d_intermediate))
W_k = np.random.normal(0, 2.0, (10, d_intermediate))
W_v = np.random.normal(0, 2.0, (10, d_intermediate))
W_q_ssf = np.random.normal(0, 0.5, (d_intermediate, d_intermediate))
W_k_ssf = np.random.normal(0, 0.5, (3, d_intermediate))
W_v_ssf = np.random.normal(0, 0.5, (3, d_intermediate))
W_out = np.random.normal(0, 0.5, (d_intermediate, 13))

def symflow_feature_fusion(jumpSeq, pc, sef, coverage_branch=0.5, coverage_path=0.5):
    """
    Generate unified 13D feature vector for SymFlow state prioritization (Sections 3.3-3.4).
    Follows demo logic with multi-dimensional attention, pre-fitted PCA, and tuned projection matrices/attention scaling.
    
    Args:
        jumpSeq (str): EVM instruction sequence (e.g., 'PUSH1 0x80 PUSH1 0x40 JUMPI').
        pc (int): Program counter (e.g., 0x12).
        sef (list): List of 10 integers [stackSize, successor, ..., subpath], normalized [0, 1].
        coverage_branch (float): SEF coverage_branch (index 3), normalized [0, 1].
        coverage_path (float): SEF coverage_path (index 4), normalized [0, 1].
    
    Returns:
        np.ndarray: Unified 13D feature vector, L2-normalized (Eq. 7).
    """
    # Input validation
    if not isinstance(jumpSeq, str):
        raise ValueError(f"jumpSeq must be a string, got {type(jumpSeq)}")
    if not isinstance(pc, int):
        raise ValueError(f"pc must be an integer, got {type(pc)}")
    sef = np.array(sef, dtype=np.float32)
    if len(sef) != 10 or not (sef >= 0).all() or not (sef <= 1).all():
        raise ValueError("SEF must be a list of 10 integers with values in [0, 1]")
    if not (0 <= coverage_branch <= 1) or not (0 <= coverage_path <= 1):
        raise ValueError("coverage_branch and coverage_path must be in [0, 1]")

    # Step 1: Create snippet dictionary (Section 3.3)
    snippet = {'jumpSeq': jumpSeq, 'pc': hex(pc)}
    input_text = f"{snippet['jumpSeq']} | pc:{snippet['pc']}"
    # print(f"输入文本: {input_text}")

    # Step 2: Generate embedding (Section 3.3)
    try:
        embedding = get_embeddings(input_text)
        # print(f"高维嵌入（前10维）: {embedding[:10].tolist()}...")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

    # Step 3: Reduce to 3D CFEF via PCA (Section 3.3)
    try:
        cfef = pca.transform(embedding.reshape(1, -1))[0]
        norm = np.linalg.norm(cfef) + 1e-8
        cfef = cfef / norm
        # print(f"CFEF向量: {cfef.tolist()}")
    except Exception as e:
        print(f"Error in PCA reduction: {e}")
        raise

    # Step 4: Feature fusion (Section 3.4)
    # print(f"\nFeature Fusion for SEF: {sef.tolist()[:5]}... CFEF: {cfef.tolist()}")
    # print(f"Coverage branch: {coverage_branch}, Coverage path: {coverage_path}")

    # Preprocessing: L2 normalize SEF and CFEF (Eq. 1-2)
    sef_scaled = sef / (np.linalg.norm(sef) + 1e-8)
    cfef_scaled = cfef
    w_coverage = min(coverage_branch + coverage_path, 2.0)
    # print(f"w_coverage: {w_coverage}")
    W_sef = np.diag([w_coverage if i in [3, 4] else 1.0 for i in range(10)])
    sef_weighted = W_sef @ sef_scaled
    # print(f"SEF weighted: {sef_weighted.tolist()[:5]}...")

    # SSI Unit (Eq. 3)
    Q_ssi = cfef_scaled @ W_q  # (8,)
    K_ssi = sef_weighted @ W_k  # (8,)
    V_ssi = sef_weighted @ W_v  # (8,)
    scores_ssi = np.clip((Q_ssi[:, None] @ K_ssi[None, :]) / np.sqrt(d_intermediate) * 4.0, -10, 10)  # (8, 8)
    attention_ssi = scipy.special.softmax(scores_ssi, axis=1)  # (8, 8)
    F_ssi = (attention_ssi @ V_ssi[:, None]).squeeze()  # (8,)
    # print(f"SSI attention (first row): {attention_ssi[0].tolist()[:5]}..., F_ssi: {F_ssi.tolist()[:5]}...")

    # SSF Unit (Eq. 4)
    Q_ssf = F_ssi @ W_q_ssf  # (8,)
    K_ssf = cfef_scaled @ W_k_ssf  # (8,)
    V_ssf = cfef_scaled @ W_v_ssf  # (8,)
    scores_ssf = np.clip((Q_ssf[:, None] @ K_ssf[None, :]) / np.sqrt(d_intermediate) * 2.0, -10, 10)  # (8, 8)
    attention_ssf = scipy.special.softmax(scores_ssf, axis=1)  # (8, 8)
    F_ssf = (attention_ssf @ V_ssf[:, None]).squeeze()  # (8,)
    # print(f"SSF attention (first row): {attention_ssf[0].tolist()[:5]}..., F_ssf: {F_ssf.tolist()[:5]}...")

    # Project to 13D
    F_ssf_projected = F_ssf @ W_out  # (13,)
    F_ssi_projected = F_ssi @ W_out  # (13,)

    # Fusion coefficient (Eq. 5)
    w_fusion = np.clip(0.3 * np.mean(attention_ssi) + 0.7 * np.mean(attention_ssf), 0.1, 0.9)

    # Preliminary fusion (Eq. 6)
    F_prelim = w_fusion * F_ssf_projected + (1 - w_fusion) * F_ssi_projected
    # print(f"F_prelim: {F_prelim.tolist()[:5]}...")

    # Residual connection and L2 normalization (Eq. 7)
    F_sef = np.concatenate([sef_weighted, np.zeros(3)])  # (13,)
    F_cfef = np.concatenate([np.zeros(10), cfef_scaled])  # (13,)
    unified_feature = F_prelim + 0.5 * w_fusion * F_sef + 0.5 * (1 - w_fusion) * F_cfef
    # print(f"Unified feature before norm: {unified_feature.tolist()[:5]}...")
    norm = np.linalg.norm(unified_feature) + 1e-8
    # print(f"Norm: {norm}")
    unified_feature = unified_feature / norm
    # print(f"Unified feature after norm: {unified_feature.tolist()[:5]}...")

    return unified_feature

# Example usage
if __name__ == "__main__":
    # Download distilroberta-base to local cache (optional)
    try:
        os.makedirs("./hf_cache", exist_ok=True)
        AutoTokenizer.from_pretrained("distilroberta-base", cache_dir="./hf_cache")
        AutoModel.from_pretrained("distilroberta-base", cache_dir="./hf_cache", use_safetensors=True)
    except Exception as e:
        print(f"Error downloading distilroberta-base to ./hf_cache: {e}")
        print("Using default cache (~/.cache/huggingface).")

    # Mock SEF data
    sef = [0.3745, 0.9507, 0.7320, 0.5987, 0.1560, 0.1560, 0.0581, 0.8662, 0.6011, 0.7081]
    unified_feature = symflow_feature_fusion(
        jumpSeq="PUSH1 0x80 PUSH1 0x40 JUMPI",
        pc=0x12,
        sef=sef,
        coverage_branch=sef[3],
        coverage_path=sef[4]
    )
    print(f"Unified Feature Vector: {unified_feature.tolist()}")
    