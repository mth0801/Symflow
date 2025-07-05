import time
import z3
from z3 import Solver, BitVec, is_bv, simplify
import sys
# base_path = '/Users/miaohuidong/opt/anaconda3/lib/python3.8'
# if base_path in sys.path:
#     sys.path.remove(base_path)
# sys.path.insert(0, '/Users/miaohuidong/opt/anaconda3/envs/se_model/lib/python3.9/site-packages')
from bse_version2 import convert_to_symbolic_bytecode, OpcodeHandlers, SymbolicVariableGenerator, SymExec, BytecodeExecutor
from testSolc import func_solc, bytecode_to_opcodes
from collections import deque
import os
import logging
from constants import STACK_MAX, SUCCESSOR_MAX, TEST_CASE_NUMBER_MAX, DEPTH_MAX, ICNT_MAX, SUBPATH_MAX, REWARD_MAX

from feature_fusion import symflow_feature_fusion

# iterLearn
# genData # trainStrategy
# SymExec dataFromTests
# update
# extractFeature predict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from bse_version2 import (
    Analysis1FunctionBodyOffChain,
    Analysis2FunctionBodyOffChain,
    Analysis3FunctionBodyOffChain,
    Analysis4FunctionBodyOffChain,
    Analysis5FunctionBodyOffChain,
)

import re

from pathlib import Path

all_single_part_execution_time = []
all_single_all_execution_time = []

# # !!! 重构
class SYMFLOWModel:
    def __init__(self, input_dim=None, hidden_dims=[128, 64], learning_rate=0.001, feature_keys=None):
        """
        初始化 LEARCH 模型
        
        Args:
            input_dim (int): 特征维度
            hidden_dims (list): 隐藏层神经元数
            learning_rate (float): 学习率
            feature_keys (list): 特征键列表
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_keys = feature_keys
        self.input_dim = input_dim if input_dim is not None else len(feature_keys) if feature_keys else None
        
        if self.input_dim is None:
            raise ValueError("input_dim or feature_keys must be provided")
        
        # 定义神经网络
        layers = []
        prev_dim = self.input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 添加 Sigmoid 激活
        self.model = nn.Sequential(*layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def prepare_data(self, dataset, test_size=0.2, random_state=42, batch_size=32):
        """
        Prepare dataset, convert to tensors, and split into training/validation sets.

        Args:
            dataset: List of tuples [(feature_list, reward), ...], where feature_list is a 13D list of floats.
            test_size: Proportion of dataset for validation (default: 0.2).
            random_state: Random seed for reproducibility (default: 42).
            batch_size: Batch size for DataLoader (default: 32).

        Returns:
            train_loader: DataLoader for training data.
            X_val: Validation feature tensor.
            y_val: Validation reward tensor.
        """
        if not dataset:
            raise ValueError("Dataset is empty")

        # Extract features and rewards
        features = np.array([d for d, _ in dataset], dtype=np.float32)  # Shape: (n_samples, 13)
        rewards = np.array([r for _, r in dataset], dtype=np.float32)   # Shape: (n_samples,)

        # Validate feature dimensions
        if features.shape[1] != 13:
            raise ValueError(f"Expected 13 features per sample, got {features.shape[1]}")

        # Validate data range
        if not (features >= -1).all() or not (features <= 1).all() or not (rewards >= 0).all() or not (rewards <= 1).all():
            raise ValueError("Features and rewards must be in [0, 1]")

        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Split into training/validation sets
        if random_state is not None:
            torch.manual_seed(random_state)
        full_dataset = TensorDataset(features, rewards)
        test_size_int = int(test_size * len(full_dataset))
        train_size = len(full_dataset) - test_size_int
        train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size_int])

        # Extract validation tensors
        X_val = torch.stack([x for x, _ in val_dataset]).to(self.device)
        y_val = torch.stack([y for _, y in val_dataset]).to(self.device)

        # Create training DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, X_val, y_val

    def train(self, train_loader, X_val, y_val, epochs=100, patience=3, min_delta=0.5e-4):
        """
        训练模型，优化早停以避免冗余轮次
        
        Args:
            train_loader: 训练数据加载器
            X_val, y_val: 验证集张量
            epochs: 最大训练轮数（默认15，基于快速收敛）
            patience: 早停耐心值（默认3，快速停止）
            min_delta: 验证损失最小改进阈值（默认1e-6，忽略微小波动）
        
        Returns:
            model: 训练好的模型
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # 验证输入数据
        if len(train_loader) == 0 or X_val.numel() == 0:
            raise ValueError("Empty training or validation data")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            
            # 批量训练
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)  # 按批量大小加权
                total_samples += batch_X.size(0)
            
            # 计算平均训练损失
            train_loss /= total_samples
            
            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()
            
            # 记录损失
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停逻辑：仅当损失显著改进时保存模型
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("best_model.pth")
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at Epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f}")
                    break
        
        # 加载最佳模型
        self.load("best_model.pth")
        logger.info("Loaded best model")
        return self.model

    def predict(self, features_1, features_2):
        """
        Predict a single reward for a state using unified features from SEF and CFEF.

        Args:
            features_1: 10D SEF feature list, e.g., [0.0, 1.0, ..., 0.0].
            features_2: CFEF pair [opcode_string, pc], e.g., ["PUSH 0x56 JUMPI", 80].

        Returns:
            reward: Float, predicted reward value.
        """
        self.model.eval()
        
        # Generate unified features
        general_features = symflow_feature_fusion(
            jumpSeq=features_2[0],
            pc=features_2[1],
            sef=features_1,
            coverage_branch=features_1[3],
            coverage_path=features_1[4]
        )
        
        # Convert to tensor
        inputs = torch.tensor(general_features, dtype=torch.float32).to(self.device)
        
        # Predict reward
        with torch.no_grad():
            reward = self.model(inputs).squeeze().cpu().item()
        
        return reward

    def save(self, path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)


class LEARCHModel:
    def __init__(self, input_dim=None, hidden_dims=[128, 64], learning_rate=0.001, feature_keys=None):
        """
        初始化 LEARCH 模型
        
        Args:
            input_dim (int): 特征维度
            hidden_dims (list): 隐藏层神经元数
            learning_rate (float): 学习率
            feature_keys (list): 特征键列表
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_keys = feature_keys
        self.input_dim = input_dim if input_dim is not None else len(feature_keys) if feature_keys else None
        
        if self.input_dim is None:
            raise ValueError("input_dim or feature_keys must be provided")
        
        # 定义神经网络
        layers = []
        prev_dim = self.input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 添加 Sigmoid 激活
        self.model = nn.Sequential(*layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def prepare_data(self, dataset, test_size=0.2, random_state=42, batch_size=32):
        """
        准备数据集，转换为张量并划分训练/验证集
        
        Args:
            dataset: 列表，[(features_dict, reward), ...]
            test_size: 验证集比例
            random_state: 随机种子
            batch_size: 批量大小
        
        Returns:
            train_loader: 训练数据加载器
            X_val, y_val: 验证集张量
        """
        if not dataset:
            raise ValueError("Dataset is empty")
        
        # 提取特征和奖励
        self.feature_keys = list(dataset[0][0].keys())  # 如 ["cpicnt", "icnt", ...]
        features = np.array([[d[key] for key in self.feature_keys] for d, _ in dataset])
        rewards = np.array([r for _, r in dataset])
        # print(features)
        # print(rewards)
        # 验证数据范围
        if not (features >= 0).all() or not (features <= 1).all() or not (rewards >= 0).all() or not (rewards <= 1).all():
            raise ValueError("Features and rewards must be in [0, 1]")
        
        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # 使用 random_split 划分训练/验证集
        if random_state is not None:
            torch.manual_seed(random_state)
        full_dataset = TensorDataset(features, rewards)
        test_size_int = int(test_size * len(full_dataset))
        train_size = len(full_dataset) - test_size_int
        train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size_int])
        
        # 提取验证集张量
        X_val = torch.stack([x for x, _ in val_dataset]).to(self.device)
        y_val = torch.stack([y for _, y in val_dataset]).to(self.device)
        
        # 创建训练数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        return train_loader, X_val, y_val

    def train(self, train_loader, X_val, y_val, epochs=100, patience=3, min_delta=0.5e-4):
        """
        训练模型，优化早停以避免冗余轮次
        
        Args:
            train_loader: 训练数据加载器
            X_val, y_val: 验证集张量
            epochs: 最大训练轮数（默认15，基于快速收敛）
            patience: 早停耐心值（默认3，快速停止）
            min_delta: 验证损失最小改进阈值（默认1e-6，忽略微小波动）
        
        Returns:
            model: 训练好的模型
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # 验证输入数据
        if len(train_loader) == 0 or X_val.numel() == 0:
            raise ValueError("Empty training or validation data")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            
            # 批量训练
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)  # 按批量大小加权
                total_samples += batch_X.size(0)
            
            # 计算平均训练损失
            train_loss /= total_samples
            
            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()
            
            # 记录损失
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停逻辑：仅当损失显著改进时保存模型
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("best_model.pth")
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at Epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f}")
                    break
        
        # 加载最佳模型
        self.load("best_model.pth")
        logger.info("Loaded best model")
        return self.model

    def predict(self, features):
        """
        预测状态奖励
        
        Args:
            features: 列表，[{features_dict}, ...]
        
        Returns:
            rewards: 预测奖励数组
        """
        self.model.eval()
        if not features:
            raise ValueError("Features list is empty")
        for f in features:
            if list(f.keys()) != self.feature_keys:
                raise ValueError(f"Feature keys mismatch: {f.keys()} vs {self.feature_keys}")
        with torch.no_grad():
            inputs = torch.tensor(
                [[f[key] for key in self.feature_keys] for f in features],
                dtype=torch.float32
            ).to(self.device)
            rewards = self.model(inputs).squeeze().cpu().numpy()
            if rewards.ndim == 0:  # Handle single input case 补丁 用于处理自动压缩的情况
                rewards = np.array([rewards.item()])  # Convert 0D to 1D
            # 移除 np.clip，因 Sigmoid 已保证输出在 (0, 1)
        return rewards

    def save(self, path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

def collect_sol_files(folder_path):
    sol_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.sol'):
            sol_files.append(os.path.join(folder_path, file))
    return sol_files

# 重构完毕
def iterLearn(way, progs, strategies, N=3): # 默认3轮学习 progs为智能合约测试集 strategies为初始启发式状态探索策略们 strategies初始化为["rss", "rps", "nurs", "sgs"]
    dataset = []
    learned = []
    for i in range(N):
        print(f"第{i + 1}轮训练开始")
        newData = genData(progs, strategies, way)
        dataset.extend(newData)

        # 新实例化一个模型类 设定好的区分方法learch symflow
        if way == "learch":
            model = LEARCHModel(input_dim=10)
        else:
            model = SYMFLOWModel(input_dim=13)
        # 准备数据
        train_loader, X_val, y_val = model.prepare_data(dataset)
        # 训练模型
        model.train(train_loader, X_val, y_val)
        if way == "learch":
            model.save(f"best_learch_model_round_{i+1}.pth")  # Round-specific
        else:
            model.save(f"best_symflow_model_round_{i+1}.pth")  # Round-specific
        learned.append(model)
        if way == "learch":
            strategies = [["learch", model]]  # 更新策略,这里是将model类的实例放进去了
        else:
            strategies = [["symflow", model]]  # 更新策略,这里是将model类的实例放进去了
    return learned

def genData(progs, strategies, way):
    dataset = []
    for strategy in strategies: # 我觉得strategy应该被设计为一个存放字段的列表,SymExec中的execute会根据字段类型进行辨认出所要使用的启发式方法并加以使用
        print(f"当前策略为{strategy}")
        i = 1
        for prog in progs:
            print(f"第{i}个.sol文件的符号执行开始")
            # 编译智能合约
            symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(prog)
            # print(len(symbolic_and_real))
            # print(symbolic_and_real)
            for item in symbolic_and_real:
                if len(item[0]) > 0 and len(item[1]) > 0:
                    executor = SymExec(item[0], item[1], strategy, way) # 需要处理一下模拟字节码和真实字节码,prog,一个执行器对应一份智能合约的处理   prog[0]存放的是模拟字节码,prog[1]存放的是真实字节码
                    success = executor.execute() # 这个应当是要重构为一个类中的方法,执行完给出测试用例tests ???
                    # 截止这里重构完毕
                    if success == False:
                        print("当前合约不具备stop或return操作码,跳过执行,直接进入下一个合约")
                        continue
                    else:
                        if way == "learch":
                            executor.prue_tree(executor.origin_node)
                            executor.count_node_reward(executor.origin_node)
                            part_dataset = build_dataset(executor.origin_node) # 找到问题所在,原先part_dataset用的是dataset,因为同名导致未归一化的数据集覆盖进去
                            normalized_dataset = normalize(part_dataset, executor)
                            print(f"normalized_dataset:{normalized_dataset}")
                            dataset.extend(normalized_dataset)
                        else:
                            # !!! 重构 参照另一条分支进行重构
                            executor.prue_tree(executor.origin_node)
                            executor.count_node_reward(executor.origin_node)
                            # symflow方式下的构建数据集(归一化也在build_dataset_symflow中做了)
                            fusioned_dataset = build_dataset_symflow(executor.origin_node, executor)
                            print(f"fusioned_dataset:{fusioned_dataset}")
                            dataset.extend(fusioned_dataset)
                else:
                    print("编译情况为空,跳过执行,直接进入下一个合约")
                    pass
            i += 1
    return dataset

def convert_runtime_opcode_to_symbolic_and_real(path):
    symbolic_and_real = []
    # symbolic_and_real = []
    with open(path,"r",) as file:
        Automata_contract = file.read()

    contracts_bytecode = func_solc(Automata_contract)
    
    for contract_id, (full_bytecode, runtime_bytecode) in contracts_bytecode.items():
        full_opcode = bytecode_to_opcodes(bytes.fromhex(full_bytecode))
        runtime_opcode = bytecode_to_opcodes(bytes.fromhex(runtime_bytecode))
        runtime_opcode_without_metadatahash = runtime_bytecode[:-88]
        runtime_opcode = bytecode_to_opcodes(
            bytes.fromhex(runtime_opcode_without_metadatahash)
        )
        symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
        symbolic_and_real.append([symbolic_bytecode, runtime_opcode])

    # contract_id = "<stdin>:ForLoopExample"  # 需要指定当前.sol的具体合约
    # contract_id = list(contracts_bytecode.keys())[0] # 默认选用当前.sol中的第一个合约
    # current_full_bytecode, current_runtime_bytecode = contracts_bytecode[contract_id]
    # runtime_opcode_without_metadatahash = current_runtime_bytecode[:-88]  # [:-88]可去除
    # runtime_opcode = bytecode_to_opcodes(
    #     bytes.fromhex(runtime_opcode_without_metadatahash)
    # )
    # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
    # symbolic_and_real.append([symbolic_bytecode, runtime_opcode])
    return symbolic_and_real
    
def bfs_read_test_tree(node, depth=0):
    indent = "    " * depth
    # stack_number={len(node.stack)} successor_number={node.successor_number} test_case_number={node.test_case_number}  depth={node.depth} cpicnt={node.cpicnt} icnt={node.icnt} covNew={node.covNew} subpath={node.subpath} branch_new_instruction={node.branch_new_instruction} path_new_instruction={node.path_new_instruction}
    print(f"{indent}- Node(index={node.bytecode_list_index} branch_new_block={node.branch_new_block} path_new_block={node.path_new_block} reward={node.reward}") # parent_node={node.parent_node} children_num={len(node.children_node)} branch_new_instruction_pc_range={node.branch_new_instruction_pc_range} executed={node.executed}
    for child in node.children_node:
        bfs_read_test_tree(child, depth + 1)
    
def count_leaf_nodes(node): # 计算测试树中没有子节点的节点（叶节点）的个数
    # 如果当前节点没有子节点，它是叶节点，计数值加 1
    if not node.children_node:
        return 1
    
    # 递归计算所有子节点的叶节点数
    leaf_count = 0
    for child in node.children_node:
        leaf_count += count_leaf_nodes(child)
    
    return leaf_count

def build_dataset_symflow(head, se):
    dataset = []

    def traverse(current_node):
        if current_node is None:
            return
        # 提取特征字典
        features_1 = [current_node.stack_size, current_node.successor_number, current_node.test_case_number, current_node.branch_new_instruction, current_node.path_new_instruction, current_node.depth, current_node.cpicnt, current_node.icnt, current_node.covNew, current_node.subpath]
        features_1 = normalize_symflow(features_1, se)
        features_2 = [current_node.jumpSeq, current_node.bytecode_list_index]
        reward = current_node.reward

        # features_1 features_2 特征融合
        general_features = symflow_feature_fusion(
            jumpSeq=features_2[0],
            pc=features_2[1],
            sef=features_1,
            coverage_branch=features_1[3],
            coverage_path=features_1[4]
        )
        dataset.append((general_features, min(reward / REWARD_MAX, 1))) # general_features 是list类型
        
        # 递归访问左右子节点
        for child in current_node.children_node:
            traverse(child)

    traverse(head)
    return dataset

def normalize_symflow(features_1, se):
    # 归一化
    features_1[0] = min(features_1[0] / STACK_MAX, 1)
    features_1[1] = min(features_1[1] / SUCCESSOR_MAX, 1)
    features_1[2] = min(features_1[2] / TEST_CASE_NUMBER_MAX, 1)
    features_1[3] = min(features_1[3] / len(se.real_bytecode), 1)
    features_1[4] = min(features_1[4] / len(se.real_bytecode), 1)
    features_1[5] = min(features_1[5] / DEPTH_MAX, 1)
    features_1[6] = min(features_1[6] / len(se.real_bytecode), 1)
    features_1[7] = min(features_1[7] / ICNT_MAX, 1)
    features_1[8] = min(features_1[8] / len(se.real_bytecode), 1)
    features_1[9] = min(features_1[9] / SUBPATH_MAX, 1)

    return features_1

def build_dataset(head):
    dataset = []

    def traverse(current_node):
        if current_node is None:
            return
        # 提取特征字典
        features = {
            "stack_size": current_node.stack_size,
            "successor_number": current_node.successor_number,
            "test_case_number": current_node.test_case_number,
            "branch_new_instruction": current_node.branch_new_instruction,
            "path_new_instruction": current_node.path_new_instruction,
            "depth": current_node.depth,
            "cpicnt": current_node.cpicnt,
            "icnt": current_node.icnt,
            "covNew": current_node.covNew,
            "subpath": current_node.subpath,
        }
        reward = current_node.reward
        dataset.append((features, reward))
        
        # 递归访问左右子节点
        for child in current_node.children_node:
            traverse(child)

    traverse(head)
    return dataset

def normalize(dataset, se): # !!!考虑归一化融入到构建数据集那一步,或者干脆融入到test记录那一步!!!
    """
    遍历数据集，将特征归一化
    
    Args:
        dataset: 列表，[(features_dict, reward), ...]
    
    Returns:
        normalized_dataset: 新数据集,已归一化
    """
    normalized_dataset = []
    for features, reward in dataset:
        # 复制特征字典
        norm_features = features.copy()
        # 归一化
        norm_features["stack_size"] = min(features["stack_size"] / STACK_MAX, 1)
        norm_features["successor_number"] = min(features["successor_number"] / SUCCESSOR_MAX, 1)
        norm_features["test_case_number"] = min(features["test_case_number"] / TEST_CASE_NUMBER_MAX, 1)
        norm_features["branch_new_instruction"] = min(features["branch_new_instruction"] / len(se.real_bytecode), 1)
        norm_features["path_new_instruction"] = min(features["path_new_instruction"] / len(se.real_bytecode), 1)
        norm_features["depth"] = min(features["depth"] / DEPTH_MAX, 1)
        norm_features["cpicnt"] = min(features["cpicnt"] / len(se.real_bytecode), 1)
        norm_features["icnt"] = min(features["icnt"] / ICNT_MAX, 1)
        norm_features["covNew"] = min(features["covNew"] / len(se.real_bytecode), 1)
        norm_features["subpath"] = min(features["subpath"] / SUBPATH_MAX, 1)
        # 归一化
        normalized_dataset.append((norm_features, min(reward / REWARD_MAX, 1))) # 找到问题所在,这里没有给出归一化后的上限1
    return normalized_dataset

# 训练好的learch模型使用于符号执行
def trained_symflow_model_use_for_se(model_path, folder_path, way):
    trained_model = SYMFLOWModel(input_dim=13)
    trained_model.load(model_path)
    # runtime_opcode = convert_func(smart_contract_path)

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[]}
    i = 1

    # designated_functions_index_range = 13  # 需要指定合约中的具体函数 13
    # is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met = (
    #     0  # 1,0 决定是否开启数据依赖关系一致性判断, 默认1开启数据依赖关系一致性判断
    # )
    # critical_state_variable_assigned_value_opcodes_number = (
    #     1  # 一开始默认至少具有一个关键状态变量赋值
    # )

    # while True:
    #     runtime_opcode, critical_state_variable_assigned_value_opcodes_number = (
    #         instantiated_main_body(
    #             runtime_opcode,
    #             designated_functions_index_range,
    #             is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met,
    #             critical_state_variable_assigned_value_opcodes_number,
    #             trained_model
    #         )
    #     )

    for sol_file in folder.glob("*.sol"):
        print(f"symflow方法下的第{i}份合约探索开始")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                executor = SymExec(item[0], item[1], ["symflow",trained_model], way) # 需要处理一下模拟字节码和真实字节码,prog,一个执行器对应一份智能合约的处理   prog[0]存放的是模拟字节码,prog[1]存放的是真实字节码
                success = executor.execute() # 这个应当是要重构为一个类中的方法,执行完给出测试用例tests ???
                if success == False:
                    print("当前合约不具备stop或return操作码,跳过执行,直接进入下一个合约")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)

                    # executor.clarify_function_information() # 进入修复工程的必经之路
            else:
                print("编译情况为空,跳过执行,直接进入下一个合约")
                pass
        i += 1
    
    print(results)

# 训练好的learch模型使用于符号执行
def trained_learch_model_use_for_se(model_path, folder_path, way):
    feature_keys = ["stack_size", "successor_number", "test_case_number", "branch_new_instruction", "path_new_instruction", "depth", "cpicnt", "icnt", "covNew", "subpath"]
    trained_model = LEARCHModel(input_dim=10, feature_keys=feature_keys)
    trained_model.load(model_path)
    # runtime_opcode = convert_func(smart_contract_path)

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[]}
    i = 1

    # designated_functions_index_range = 13  # 需要指定合约中的具体函数 13
    # is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met = (
    #     0  # 1,0 决定是否开启数据依赖关系一致性判断, 默认1开启数据依赖关系一致性判断
    # )
    # critical_state_variable_assigned_value_opcodes_number = (
    #     1  # 一开始默认至少具有一个关键状态变量赋值
    # )

    # while True:
    #     runtime_opcode, critical_state_variable_assigned_value_opcodes_number = (
    #         instantiated_main_body(
    #             runtime_opcode,
    #             designated_functions_index_range,
    #             is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met,
    #             critical_state_variable_assigned_value_opcodes_number,
    #             trained_model
    #         )
    #     )

    for sol_file in folder.glob("*.sol"):
        print(f"learch方法下的第{i}份合约探索开始")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                executor = SymExec(item[0], item[1], ["learch",trained_model], way) # 需要处理一下模拟字节码和真实字节码,prog,一个执行器对应一份智能合约的处理   prog[0]存放的是模拟字节码,prog[1]存放的是真实字节码
                success = executor.execute() # 这个应当是要重构为一个类中的方法,执行完给出测试用例tests ???
                if success == False:
                    print("当前合约不具备stop或return操作码,跳过执行,直接进入下一个合约")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)

                    # executor.clarify_function_information() # 进入修复工程的必经之路
            else:
                print("编译情况为空,跳过执行,直接进入下一个合约")
                pass
        i += 1
    
    print(results)

# rss使用于符号执行
def rss_use_for_se(folder_path, way):
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"coverage":[], "select_state_accuracy":[], "arrive_assigned_coverage_time":[]}
    i = 1
    for sol_file in folder.glob("*.sol"):
        print(f"rss方法下的第{i}份合约探索开始")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")

        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                executor = SymExec(item[0], item[1], "rss", way) # 需要处理一下模拟字节码和真实字节码,prog,一个执行器对应一份智能合约的处理   prog[0]存放的是模拟字节码,prog[1]存放的是真实字节码
                success = executor.execute() # 这个应当是要重构为一个类中的方法,执行完给出测试用例tests ???
                if success == False:
                    print("当前合约不具备stop或return操作码,跳过执行,直接进入下一个合约")
                else:
                    results["coverage"].append(executor.coverage)
                    results["select_state_accuracy"].append(executor.select_state_accuracy)
                    if (executor.arrive_assigned_coverage_time != []):
                        results["arrive_assigned_coverage_time"].append(executor.arrive_assigned_coverage_time[0])
                    else:
                        results["arrive_assigned_coverage_time"].append(None)

                    # print(f"executor.coverage:{executor.coverage}")
                    # print(f"executor.select_state_accuracy:{executor.select_state_accuracy}")
                    # print(f"executor.arrive_assigned_coverage_time:{executor.arrive_assigned_coverage_time}")
                    
                    # print(len(executor.control_flow_graph))
                    # print(count_smart_contract_jump_jumpi_number(executor.real_bytecode))
                    
                    # executor.clarify_function_information() # 进入修复工程的必经之路

            else:
                print("编译情况为空,跳过执行,直接进入下一个合约")
                pass
        i += 1
    print(results)

def count_sm_bytecode_len(folder_path, way):

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return {}

    results = {"bytecode_len":[]}
    i = 1

    for sol_file in folder.glob("*.sol"):
        print(f"第{i}份合约的bytecode长度统计开始")
        smart_contract_path = str(sol_file)
        print(f"smart_contract_path:{smart_contract_path}")
        
        symbolic_and_real = convert_runtime_opcode_to_symbolic_and_real(smart_contract_path)
        # runtime_opcode = convert_func(smart_contract_path)
        # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)

        for item in symbolic_and_real:
            if len(item[0]) > 0 and len(item[1]) > 0:
                executor = SymExec(item[0], item[1], "rss", way) # 需要处理一下模拟字节码和真实字节码,prog,一个执行器对应一份智能合约的处理   prog[0]存放的是模拟字节码,prog[1]存放的是真实字节码
                success = executor.execute() # 这个应当是要重构为一个类中的方法,执行完给出测试用例tests ???
                if success == False:
                    print("当前合约不具备stop或return操作码,跳过执行,直接进入下一个合约")
                else:
                    results["bytecode_len"].append(len(executor.real_bytecode))
                    print(f"bytecode_len:{len(executor.real_bytecode)}")
            else:
                print("编译情况为空,跳过执行,直接进入下一个合约")
                pass
        i += 1
    
    print(results)


# 训练好的模型使用于预测
def trained_model_use_for_predict(model_path, feature_demo):
    feature_keys = ["stack_size", "successor_number", "test_case_number", "branch_new_instruction", "path_new_instruction", "depth", "cpicnt", "icnt", "covNew", "subpath"]
    trained_model = LEARCHModel(input_dim=10, feature_keys=feature_keys)
    trained_model.load(model_path)
    predicted_rewards = trained_model.predict(feature_demo)
    print(f"Predicted Rewards: {predicted_rewards}")


def convert_func(path): # 指定具体合约再将运行时字节码转换为操作码相关 (截取部分)
    # symbolic_and_real = []
    with open(path,"r",) as file:
        Automata_contract = file.read()

    contracts_bytecode = func_solc(Automata_contract)

    # contract_id = "<stdin>:DAO"  # 需要指定当前.sol的具体合约 _50Win
    contract_id = list(contracts_bytecode.keys())[0] # 默认选用当前.sol中的第一个合约

    current_full_bytecode, current_runtime_bytecode = contracts_bytecode[contract_id]
    runtime_opcode_without_metadatahash = current_runtime_bytecode[:-88]  # [:-88]可去除
    runtime_opcode = bytecode_to_opcodes(
        bytes.fromhex(runtime_opcode_without_metadatahash)
    )
    # symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
    # symbolic_and_real.append([symbolic_bytecode, runtime_opcode])
    # return symbolic_and_real
    return runtime_opcode

def main():
    sol_files = collect_sol_files("/Users/miaohuidong/demos/RESC/test_smartcontract_dataset/dataset_for_train") # dataset_for_train
    # iterLearn("learch", sol_files, ["rss"])
    iterLearn("symflow", sol_files, ["rss"])
    

# def main():

#     progs = convert_runtime_opcode_to_symbolic_and_real()
#     se = SymExec(progs[0][0], progs[0][1], "rss")
#     all_stacks = se.execute()
    
#     print(f"se.control_flow_graph:{se.control_flow_graph}")
#     # machine_learning_for_se.iterLearn(progs, ) # 策略待填充 ???
#     bfs_read_test_tree(se.origin_node) # 遍历树
#     print(f"未剪枝前的叶子节点数:{count_leaf_nodes(se.origin_node)}")
#     print(f"se.passed_program_paths:{se.passed_program_paths}")
#     for item in se.passed_program_paths:
#         print(se.real_bytecode[item[-1]])
#     # 未处理的叶节点应该被剪枝
#     print(f"se.passed_program_paths_to_passed_number:{se.passed_program_paths_to_passed_number}")

#     print(f"se.subpath_k4_to_number:{se.subpath_k4_to_number}")

#     se.prue_tree(se.origin_node) # 去除无用节点,剪枝

#     print(f"头节点的奖励值:{se.count_node_reward(se.origin_node)}") # 计算每个节点的奖励,目前仅考虑探索新基本块因素,不考虑时间因素

#     bfs_read_test_tree(se.origin_node) # 遍历树
#     print(f"剪枝后的叶子节点数:{count_leaf_nodes(se.origin_node)}")
#     print(f"test_case_num总数:{se.test_case_num}")
#     # test_case_num总数和剪枝后的叶子节点数不相等的原因在于:循环的部分并不会主动计入test_case_num，而在抵达时间阈值的时候，这些循环部分又成为有效的叶子节点，因此循环的部分就是多出来的叶子节点

#     dataset = build_dataset(se.origin_node)
#     print(f"dataset:{dataset}")
#     print(f"len(dataset):{len(dataset)}")

#     normalized_dataset = normalize(dataset, se)
#     print(f"normalized_dataset:{normalized_dataset}")

#     # se = SymExec(progs[0][0], progs[0][1], "learned")

#     # se.clarify_function_information() # 进入修复工程的必经之路,此处暂时注释

# # 示例使用
# def demo_train():
#     # 模拟数据集（替换为你的实际数据集）
#     dataset = 
#     # 初始化模型
#     model = LEARCHModel(input_dim=10)  # 10 个特征
#     # 准备数据
#     train_loader, X_val, y_val = model.prepare_data(dataset)
#     # 训练模型
#     model.train(train_loader, X_val, y_val)
#     # 预测示例         
#     test_features = [{'stack_size': 0.02, 'successor_number': 0.0, 'test_case_number': 0.0, 'branch_new_instruction': 0.021834061135371178, 'path_new_instruction': 0.12663755458515283, 'depth': 0.1, 'cpicnt': 0.12663755458515283, 'icnt': 0.0, 'covNew': 0.0, 'subpath': 0.0}, {'stack_size': 0.02, 'successor_number': 0.0, 'test_case_number': 0.0, 'branch_new_instruction': 0.021834061135371178, 'path_new_instruction': 0.12663755458515283, 'depth': 0.1, 'cpicnt': 0.12663755458515283, 'icnt': 0.0, 'covNew': 0.0, 'subpath': 0.0}]
#     predicted_rewards = model.predict(test_features)
#     print(f"Predicted Rewards: {predicted_rewards}")

def instantiated_main_body(
    runtime_opcode,
    designated_functions_index_range,
    is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met,
    critical_state_variable_assigned_value_opcodes_number,
    trained_model
):
    if critical_state_variable_assigned_value_opcodes_number == 0:
        raise UnboundLocalError("The function has finished its reorder!")
    else:
        print(f"current runtime_opcode: {runtime_opcode}")
        symbolic_bytecode = convert_to_symbolic_bytecode(runtime_opcode)
        # 执行符号字节码
        print(f"symbolic_bytecode:{symbolic_bytecode}")
        # executor = SymbolicBytecodeExecutor(symbolic_bytecode, runtime_opcode)
        executor = SymExec(symbolic_bytecode, runtime_opcode, trained_model) # 待填充way !!!

        all_single_all_execution_time.append(time.time())
        print(f"all_single_all_execution_time: {all_single_all_execution_time}")

        success = executor.execute()
        # print(all_stacks)  # 输出符号表达式堆栈，例如: [v0 + v1 - v2, pc, msize, gas]

        executor.clarify_function_information() # 进入修复工程的必经之路

        # function_start_index = executor.get_max_stop_return_index()
        # print(f"function_start_index: {function_start_index}")

        # # 在模拟执行完毕后调用
        # cfg = executor.create_control_flow_graph()
        # print(f"cfg: {cfg}")
        # print(type(cfg))
        # # cfg.render('control_flow_graph', format='png')  # 保存为 PNG 文件
        # # cfg.view()  # 显示控制流图

        # print(executor.visited_nodes_index_by_jumpi)
        # print(type(executor.visited_nodes_index_by_jumpi))
        # print(executor.exist_loop_node_by_jumpi)
        # print(f"executor.exist_loop_node_by_jumpi: {executor.exist_loop_node_by_jumpi}")

        # # 查询指定索引位置的操作码的PC位置
        # test_index = 17  # 替换为你需要查询的索引位置
        # pc_position = executor.get_pc_position(test_index)
        # print(f"索引位置 {test_index} 对应的PC位置是: {pc_position}")

        # # 查询指定PC位置的操作码的索引位置
        # test_pc = 46  # 替换为你需要查询的索引位置
        # index_position = executor.get_index_position(test_pc)
        # print(f"PC位置 {test_pc} 对应的索引位置是: {index_position}")

        executor.stack_snapshots = dict(sorted(executor.stack_snapshots.items()))
        print(f"executor.stack_snapshots: {executor.stack_snapshots}")
        # print(f"executor.opcodeindex_to_stack: {executor.opcodeindex_to_stack}")  # ?
        print(
            f"executor.smartcontract_functions_index_range: {executor.smartcontract_functions_index_range}"
        )

        # # print(executor.stack_snapshots[232])
        # # print(executor.stack_snapshots[241])
        # # print(executor.stack_snapshots[276])

        all_single_part_execution_time.append(time.time())
        print(f"all_single_part_execution_time: {all_single_part_execution_time}")

        analysis1_function_body_off_chain_machine = Analysis1FunctionBodyOffChain(
            executor
        )
        temporary_variable_quantity, take_special_stack_snapshots_index = (
            analysis1_function_body_off_chain_machine.count_consecutive_push_0_push_60_dup1(
                designated_functions_index_range
            )
        )
        print(f"temporary_variable_quantity: {temporary_variable_quantity}")
        print(
            f"take_special_stack_snapshots_index: {take_special_stack_snapshots_index}"
        )
        # 现在跳转地址是能够具体确定的，包括显式和隐式都可以确定，我们采用的是获取堆栈顶部操作数的方法来确定的

        analysis2_function_body_off_chain_machine = Analysis2FunctionBodyOffChain(
            executor,
            take_special_stack_snapshots_index,
            designated_functions_index_range,
        )
        old_jump_structure_info = analysis2_function_body_off_chain_machine.traverse_designated_function_bytecode(
            executor.smartcontract_functions_index_range[
                designated_functions_index_range
            ],
            executor.smartcontract_functions_index_range[
                designated_functions_index_range
            ],
            current_jump_depth=0,
        )
        print(f"old_jump_structure_info: {old_jump_structure_info}")

        analysis3_function_body_off_chain_machine = Analysis3FunctionBodyOffChain(
            executor,
            take_special_stack_snapshots_index,
            designated_functions_index_range,
            old_jump_structure_info,
        )
        new_jump_structure_info = (
            analysis3_function_body_off_chain_machine.bytecode_ByteDance_granularity_segmentation_by_jump_depth()
        )

        analysis4_function_body_off_chain_machine = Analysis4FunctionBodyOffChain(
            executor,
            take_special_stack_snapshots_index,
            designated_functions_index_range,
            new_jump_structure_info,
            temporary_variable_quantity,
        )

        all_create_opcodes_index_list = (
            analysis4_function_body_off_chain_machine.search_all_create_opcode()
        )  # 当前函数内的所有CREATE操作码的index位置
        print(f"all_create_opcodes_index_list: {all_create_opcodes_index_list}")

        transfer_accounts_opcodes_index_list = (
            analysis4_function_body_off_chain_machine.search_transfer_accounts_opcode(
                executor.smartcontract_functions_index_range[
                    designated_functions_index_range
                ][0],
                executor.smartcontract_functions_index_range[
                    designated_functions_index_range
                ][1],
            )
        )  # 关键CALL操作码的index位置

        print(
            f"transfer_accounts_opcodes_index_list: {transfer_accounts_opcodes_index_list}"
        )

        # all_single_part_execution_time.append(time.time())
        # print(f"all_single_part_execution_time: {all_single_part_execution_time}")
        # all_single_all_execution_time.append(time.time())
        # print(f"all_single_all_execution_time: {all_single_all_execution_time}")

        if len(transfer_accounts_opcodes_index_list) == 0:
            raise UnboundLocalError("The function does not have transfer money block!")

        # critical_state_variable_assigned_value_opcodes_index_list = (
        #     analysis4_function_body_off_chain_machine.search_critical_state_variable_assigned_value_opcode_by_critical_branch_jump_structure()
        # )  # 当前函数内的关键SSTORE操作码的index位置
        critical_state_variable_assigned_value_opcodes_index_list = (
            analysis4_function_body_off_chain_machine.search_all_critical_state_variable_assigned_value_opcode()
        )  # 当前函数内的转帐之后的所有SSTORE操作码的index位置

        all_single_part_execution_time.append(time.time())
        print(f"all_single_part_execution_time: {all_single_part_execution_time}")
        all_single_all_execution_time.append(time.time())
        print(f"all_single_all_execution_time: {all_single_all_execution_time}")

        if len(critical_state_variable_assigned_value_opcodes_index_list) == 0:
            raise UnboundLocalError(
                "The function does not have status variables assignment block!"
            )

        final_jump_structure1, final_jump_structure2 = (
            analysis4_function_body_off_chain_machine.search_parent_jump_structure_in_the_same_deepest_detecting_range()
        )
        if analysis4_function_body_off_chain_machine.step1_can_reorder_or_not:
            # 先检查关键状态变量赋值块中的被赋值项和传播项
            (
                critical_propagation_items,
                critical_assigned_items,
                index_mapping_to_critical_propagation_items,
                index_mapping_to_critical_assigned_items,
            ) = analysis4_function_body_off_chain_machine.record_critical_propagation_items_and_assigned_items(
                analysis4_function_body_off_chain_machine.final_jump_structure2[
                    "jump_structure_index_range"
                ][0],
                analysis4_function_body_off_chain_machine.final_jump_structure2[
                    "jump_structure_index_range"
                ][1],
            )
            print(f"critical_propagation_items: {critical_propagation_items}")
            print(f"critical_assigned_items: {critical_assigned_items}")

            # 再检查中间部分中的被赋值项和传播项
            (
                middle_propagation_items,
                middle_assigned_items,
                index_mapping_to_middle_propagation_items,
                index_mapping_to_middle_assigned_items,
            ) = analysis4_function_body_off_chain_machine.record_middle_propagation_items_and_assigned_items(
                analysis4_function_body_off_chain_machine.final_jump_structure1[
                    "jump_structure_index_range"
                ][0],
                analysis4_function_body_off_chain_machine.final_jump_structure2[
                    "jump_structure_index_range"
                ][0],
            )
            print(f"middle_propagation_items: {middle_propagation_items}")
            print(f"middle_assigned_items: {middle_assigned_items}")

            analysis5_function_body_off_chain_machine = Analysis5FunctionBodyOffChain(
                executor,
                take_special_stack_snapshots_index,
                designated_functions_index_range,
                new_jump_structure_info,
                temporary_variable_quantity,
                critical_propagation_items,
                critical_assigned_items,
                middle_propagation_items,
                middle_assigned_items,
                final_jump_structure1,
                final_jump_structure2,
                index_mapping_to_critical_propagation_items,
                index_mapping_to_critical_assigned_items,
                index_mapping_to_middle_propagation_items,
                index_mapping_to_middle_assigned_items,
            )
            step2_can_reorder_or_not = analysis5_function_body_off_chain_machine.determine_whether_the_consistency_of_data_dependency_relationships_is_met(
                is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met
            )
            if step2_can_reorder_or_not:
                adjacent_or_same_signal = 1  # 相同深度
                reorder_function_real_bytecode, reorder_real_bytecode = (
                    analysis5_function_body_off_chain_machine.reorder_key_granularity_bytecode_blocks(
                        adjacent_or_same_signal
                    )
                )

                with open(
                    "/Users/miaohuidong/demos/RESC/test_txt/bytecode1.txt", "w"
                ) as f:
                    for opcode in reorder_real_bytecode:
                        f.write(opcode + "\n")
                print("reorder bytecode has been written into bytecode1.txt")

            else:

                all_single_part_execution_time.append(time.time())
                print(
                    f"all_single_part_execution_time: {all_single_part_execution_time}"
                )
                all_single_all_execution_time.append(time.time())
                print(f"all_single_all_execution_time: {all_single_all_execution_time}")
                # print(symbolic_bytecode[862:905])
                # print(symbolic_bytecode[959:987])
                raise UnboundLocalError("step2_can_reorder_or_not is False!")
                # 待扩展
        else:
            final_jump_structure1, final_jump_structure2 = (
                analysis4_function_body_off_chain_machine.search_parent_jump_structure_in_the_adjacent_deepest_detecting_range()
            )
            if analysis4_function_body_off_chain_machine.step1_can_reorder_or_not:
                # 先检查关键状态变量赋值块中的被赋值项和传播项
                (
                    critical_propagation_items,
                    critical_assigned_items,
                    index_mapping_to_critical_propagation_items,
                    index_mapping_to_critical_assigned_items,
                ) = analysis4_function_body_off_chain_machine.record_critical_propagation_items_and_assigned_items(
                    analysis4_function_body_off_chain_machine.final_jump_structure2[
                        "jump_structure_index_range"
                    ][0],
                    analysis4_function_body_off_chain_machine.final_jump_structure2[
                        "jump_structure_index_range"
                    ][1],
                )
                print(f"critical_propagation_items: {critical_propagation_items}")
                print(f"critical_assigned_items: {critical_assigned_items}")

                # 再检查中间部分中的被赋值项和传播项
                (
                    middle_propagation_items,
                    middle_assigned_items,
                    index_mapping_to_middle_propagation_items,
                    index_mapping_to_middle_assigned_items,
                ) = analysis4_function_body_off_chain_machine.record_middle_propagation_items_and_assigned_items(
                    analysis4_function_body_off_chain_machine.final_jump_structure1[
                        "jump_structure_index_range"
                    ][0],
                    analysis4_function_body_off_chain_machine.final_jump_structure2[
                        "jump_structure_index_range"
                    ][0],
                )
                print(f"middle_propagation_items: {middle_propagation_items}")
                print(f"middle_assigned_items: {middle_assigned_items}")

                analysis5_function_body_off_chain_machine = (
                    Analysis5FunctionBodyOffChain(
                        executor,
                        take_special_stack_snapshots_index,
                        designated_functions_index_range,
                        new_jump_structure_info,
                        temporary_variable_quantity,
                        critical_propagation_items,
                        critical_assigned_items,
                        middle_propagation_items,
                        middle_assigned_items,
                        final_jump_structure1,
                        final_jump_structure2,
                        index_mapping_to_critical_propagation_items,
                        index_mapping_to_critical_assigned_items,
                        index_mapping_to_middle_propagation_items,
                        index_mapping_to_middle_assigned_items,
                    )
                )
                step2_can_reorder_or_not = analysis5_function_body_off_chain_machine.determine_whether_the_consistency_of_data_dependency_relationships_is_met(
                    is_or_not_determine_whether_the_consistency_of_data_dependency_relationships_is_met
                )
                if step2_can_reorder_or_not:
                    adjacent_or_same_signal = 0  # 相邻深度
                    reorder_function_real_bytecode, reorder_real_bytecode = (
                        analysis5_function_body_off_chain_machine.reorder_key_granularity_bytecode_blocks(
                            adjacent_or_same_signal
                        )
                    )

                    with open(
                        "/Users/miaohuidong/demos/RESC/test_txt/bytecode1.txt", "w"
                    ) as f:
                        for opcode in reorder_real_bytecode:
                            f.write(opcode + "\n")
                    print("reorder bytecode has been written into bytecode1.txt")

                else:

                    all_single_part_execution_time.append(time.time())
                    print(
                        f"all_single_part_execution_time: {all_single_part_execution_time}"
                    )
                    all_single_all_execution_time.append(time.time())
                    print(
                        f"all_single_all_execution_time: {all_single_all_execution_time}"
                    )

                    raise UnboundLocalError("step2_can_reorder_or_not is False!")
                    # 待扩展
            else:
                all_single_part_execution_time.append(time.time())
                print(
                    f"all_single_part_execution_time: {all_single_part_execution_time}"
                )
                all_single_all_execution_time.append(time.time())
                print(f"all_single_all_execution_time: {all_single_all_execution_time}")

                raise UnboundLocalError("step1_can_reorder_or_not is False!")

        all_single_part_execution_time.append(time.time())
        print(f"all_single_part_execution_time: {all_single_part_execution_time}")
        all_single_all_execution_time.append(time.time())
        print(f"all_single_all_execution_time: {all_single_all_execution_time}")

    return (
        reorder_real_bytecode,
        len(critical_state_variable_assigned_value_opcodes_index_list) - 1,
    )


if __name__ == "__main__":
    # demo_train()

    # main() # 训练model

    # 实验测试基线和symflow
    # trained_symflow_model_use_for_se("/Users/miaohuidong/demos/RESC/trained_symflow_model/best_symflow_model_round_3.pth", "/Users/miaohuidong/demos/RESC/test_smartcontract_dataset/test_dataset_for_train", "symflow")
    # trained_learch_model_use_for_se("/Users/miaohuidong/demos/RESC/trained_learch_model/best_model_round_3.pth", "/Users/miaohuidong/demos/RESC/test_smartcontract_dataset/test_dataset_for_train", "learch")
    # rss_use_for_se("/Users/miaohuidong/demos/RESC/test_smartcontract_dataset/test_dataset_for_train", "learch")
    
    # 统计所有test智能合约的字节码长度
    count_sm_bytecode_len("/Users/miaohuidong/demos/RESC/test_smartcontract_dataset/test_dataset_for_train", "learch")

    # trained_model_use_for_predict("/Users/miaohuidong/demos/RESC/trained_model/best_model_round_3.pth", [{'stack_size': 0.16666666666666666, 'successor_number': 0.5, 'test_case_number': 0.8, 'branch_new_instruction': 0.0, 'path_new_instruction': 0.0, 'depth': 1, 'cpicnt': 0.1371976647206005, 'icnt': 0.2, 'covNew': 0.014595496246872394, 'subpath': 0.4}])
