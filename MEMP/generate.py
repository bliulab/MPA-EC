import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# 在文件开头导入sklearn相关模块
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
import joblib

# ===================== Rotary Embedding =====================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return self.cos_cached[:, :, :x.size(1), :], self.sin_cached[:, :, :x.size(1), :]

    def rotate_queries_or_keys(self, x):
        cos, sin = self(x)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

# ===================== 特征提取器 =====================

class ProteinFeatureExtractor:
    """蛋白质序列特征提取器"""
    def __init__(self):
        # 氨基酸理化性质
        self.aa_properties = {
            'A': {'hydrophobic': 1.8, 'volume': 88.6, 'pka': 2.34, 'charge': 0, 'polar': 0},
            'R': {'hydrophobic': -4.5, 'volume': 173.4, 'pka': 12.48, 'charge': 1, 'polar': 1},
            'N': {'hydrophobic': -3.5, 'volume': 114.1, 'pka': 2.02, 'charge': 0, 'polar': 1},
            'D': {'hydrophobic': -3.5, 'volume': 111.1, 'pka': 3.65, 'charge': -1, 'polar': 1},
            'C': {'hydrophobic': 2.5, 'volume': 108.5, 'pka': 1.96, 'charge': 0, 'polar': 0},
            'Q': {'hydrophobic': -3.5, 'volume': 143.8, 'pka': 2.17, 'charge': 0, 'polar': 1},
            'E': {'hydrophobic': -3.5, 'volume': 138.4, 'pka': 4.25, 'charge': -1, 'polar': 1},
            'G': {'hydrophobic': -0.4, 'volume': 60.1, 'pka': 2.34, 'charge': 0, 'polar': 0},
            'H': {'hydrophobic': -3.2, 'volume': 153.2, 'pka': 6.00, 'charge': 0.1, 'polar': 1},
            'I': {'hydrophobic': 4.5, 'volume': 166.7, 'pka': 2.36, 'charge': 0, 'polar': 0},
            'L': {'hydrophobic': 3.8, 'volume': 166.7, 'pka': 2.36, 'charge': 0, 'polar': 0},
            'K': {'hydrophobic': -3.9, 'volume': 168.6, 'pka': 10.53, 'charge': 1, 'polar': 1},
            'M': {'hydrophobic': 1.9, 'volume': 162.9, 'pka': 2.28, 'charge': 0, 'polar': 0},
            'F': {'hydrophobic': 2.8, 'volume': 189.9, 'pka': 1.83, 'charge': 0, 'polar': 0},
            'P': {'hydrophobic': -1.6, 'volume': 112.7, 'pka': 1.99, 'charge': 0, 'polar': 0},
            'S': {'hydrophobic': -0.8, 'volume': 89.0, 'pka': 2.21, 'charge': 0, 'polar': 1},
            'T': {'hydrophobic': -0.7, 'volume': 116.1, 'pka': 2.09, 'charge': 0, 'polar': 1},
            'W': {'hydrophobic': -0.9, 'volume': 227.8, 'pka': 2.83, 'charge': 0, 'polar': 0},
            'Y': {'hydrophobic': -1.3, 'volume': 193.6, 'pka': 2.20, 'charge': 0, 'polar': 1},
            'V': {'hydrophobic': 4.2, 'volume': 140.0, 'pka': 2.32, 'charge': 0, 'polar': 0}
        }
    
    def extract_features(self, sequence):
        """提取蛋白质序列特征"""
        if not sequence or len(sequence) == 0:
            return np.zeros(46)
        
        features = []
        features.append(len(sequence))
        
        aa_counts = {aa: 0 for aa in self.aa_properties}
        for aa in sequence:
            if aa in aa_counts:
                aa_counts[aa] += 1
        
        aa_freqs = {aa: count/len(sequence) for aa, count in aa_counts.items()}
        features.extend(aa_freqs.values())
        
        properties = ['hydrophobic', 'volume', 'pka', 'charge', 'polar']
        for prop in properties:
            values = [self.aa_properties[aa][prop] for aa in sequence if aa in self.aa_properties]
            if values:
                features.extend([
                    np.mean(values), np.std(values), np.min(values),
                    np.max(values), np.median(values)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        features = np.array(features[:46])
        if len(features) < 46:
            features = np.pad(features, (0, 46 - len(features)), 'constant')
        
        return features

# ===================== 机器学习模型类 =====================

class MLModelEnsemble:
    """机器学习模型集成"""
    def __init__(self, task_name, random_state=42):
        self.task_name = task_name
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.is_fitted = {}
        
        if task_name == 'ph':
            self.model_configs = {
                'svr_rbf': SVR(kernel='rbf', gamma='auto', C=20, epsilon=0.05),
                'rf': RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=2, random_state=random_state),
                'extra_trees': ExtraTreesRegressor(n_estimators=200, max_depth=25, random_state=random_state),
                'gbr': GradientBoostingRegressor(n_estimators=200, max_depth=10, learning_rate=0.08, random_state=random_state),
                'mlp': MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=random_state)
            }
        elif task_name == 'tem':
            self.model_configs = {
                'svr_rbf': SVR(kernel='rbf', gamma='auto', C=200, epsilon=0.8),
                'rf': RandomForestRegressor(n_estimators=400, max_depth=25, min_samples_split=2, random_state=random_state),
                'extra_trees': ExtraTreesRegressor(n_estimators=300, max_depth=30, random_state=random_state),
                'gbr': GradientBoostingRegressor(n_estimators=300, max_depth=12, learning_rate=0.03, random_state=random_state),
                'mlp': MLPRegressor(hidden_layer_sizes=(300, 150, 75), max_iter=1000, random_state=random_state)
            }
        else:
            self.model_configs = {
                'svr_rbf': SVR(kernel='rbf', gamma='auto', C=100, epsilon=0.5),
                'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state),
                'gbr': GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.08, random_state=random_state)
            }
        
        for name, model in self.model_configs.items():
            self.models[name] = model
            self.scalers[name] = RobustScaler() if task_name == 'ph' else StandardScaler()
            self.is_fitted[name] = False
    
    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            print(f"Warning: No data for {self.task_name} ML training")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training ML models for {self.task_name} with {len(X)} samples...")
        
        for name, model in self.models.items():
            try:
                X_scaled = self.scalers[name].fit_transform(X)
                model.fit(X_scaled, y)
                self.is_fitted[name] = True
                
                train_pred = model.predict(X_scaled)
                train_r2 = r2_score(y, train_pred)
                print(f"  {name}: Training R² = {train_r2:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                self.is_fitted[name] = False
    
    def predict(self, X):
        if len(X) == 0:
            return np.array([])
        
        X = np.array(X)
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if self.is_fitted[name]:
                try:
                    X_scaled = self.scalers[name].transform(X)
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                    
                    if 'rf' in name:
                        weights.append(1.1)
                    elif name == 'gbr':
                        weights.append(1.2)
                    else:
                        weights.append(1.0)
                        
                except Exception as e:
                    print(f"Prediction error for {name}: {e}")
        
        if predictions:
            predictions = np.array(predictions)
            weights = np.array(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
        else:
            return np.zeros(len(X))
    
    def save(self, checkpoint_dir):
        model_dir = os.path.join(checkpoint_dir, f"ml_models_{self.task_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if self.is_fitted[name]:
                joblib.dump(model, os.path.join(model_dir, f"{name}_model.pkl"))
                joblib.dump(self.scalers[name], os.path.join(model_dir, f"{name}_scaler.pkl"))
        
        joblib.dump(self.is_fitted, os.path.join(model_dir, "fitted_status.pkl"))
    
    def load(self, checkpoint_dir):
        model_dir = os.path.join(checkpoint_dir, f"ml_models_{self.task_name}")
        if not os.path.exists(model_dir):
            return False
        
        try:
            fitted_status = joblib.load(os.path.join(model_dir, "fitted_status.pkl"))
            
            for name in self.models:
                if fitted_status.get(name, False):
                    self.models[name] = joblib.load(os.path.join(model_dir, f"{name}_model.pkl"))
                    self.scalers[name] = joblib.load(os.path.join(model_dir, f"{name}_scaler.pkl"))
                    self.is_fitted[name] = True
            
            return True
        except Exception as e:
            print(f"Error loading ML models for {self.task_name}: {e}")
            return False

# ===================== 简化但有效的缓存系统 =====================

class SimpleSequenceCache:
    """简化但有效的序列缓存系统"""
    def __init__(self, model, tokenizer, device):
        self.cache = {}
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.model_max_length = int(1e9)
        self.hidden_size = model.config.hidden_size
        
        print(f"Simple cache - Model hidden size: {self.hidden_size}")
    
    def __call__(self, sequence):
        if sequence not in self.cache:
            tokens = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                padding=True, 
                truncation=False
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                embeds = self.model.get_input_embeddings()(tokens)
                
                # 使用简单的平均池化
                mean_pool = embeds.mean(dim=1)
                
                self.cache[sequence] = mean_pool.cpu()
        
        return self.cache[sequence].to(self.device)

# ===================== MLP模块 =====================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout)
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

# ===================== 残差块 =====================

class ResBlock1D(nn.Module):
    """1-D 残差块：Conv1d → BN → GELU → Conv1d → BN"""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))

# ===================== 简化但有效的神经网络架构 =====================

class SimpleProteinPredictor(nn.Module):
    """简化但有效的蛋白质预测器"""
    def __init__(self, input_dim=4096, hidden_dim=2048, dropout=0.1, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        
        print(f"Creating SimpleProteinPredictor with input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        # 输入处理
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
        )
        
        # 任务特定的预测头
        self.ph_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.tem_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.ogt_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.tm_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Simple model created with {total_params:,} trainable parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 3:
            x = x.squeeze(1)
        
        if x.size(1) != self.input_dim:
            raise ValueError(f"Input dim mismatch: got {x.size(1)}, expected {self.input_dim}")
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # 输入处理
        x = self.input_processor(x)
        
        # 任务预测
        ph_raw = self.ph_head(x).squeeze(-1)
        ph_pred = ph_raw * 14.0
        
        tem_pred = self.tem_head(x).squeeze(-1)
        ogt_pred = self.ogt_head(x).squeeze(-1)
        tm_pred = self.tm_head(x).squeeze(-1)
        
        # 范围约束
        ph_pred = torch.clamp(ph_pred, 0, 14)
        tem_pred = torch.clamp(tem_pred, 0, 150)
        ogt_pred = torch.clamp(ogt_pred, 0, 120)
        tm_pred = torch.clamp(tm_pred, 0, 100)
        
        return ph_pred, tem_pred, ogt_pred, tm_pred

# ===================== RandomForest模块 =====================

class RandomForest(nn.Module):
    def __init__(self,
                 input_dim: int = 4096,
                 hidden_dim: int = 256,
                 num_blocks: int = 4,
                 dilations: tuple = (1, 2, 4, 8),
                 device: str = "cuda"):
        super().__init__()
        self.device  = torch.device(device)
        self.dtype   = torch.float32

        # 1×1 降维
        self.proj = nn.Conv1d(input_dim, hidden_dim, 1).to(dtype=self.dtype)

        # 残差瓶颈：多尺度膨胀卷积 + GN + SiLU
        self.blocks = nn.ModuleList()
        for d in dilations[:num_blocks]:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=d, dilation=d, groups=hidden_dim),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(inplace=False),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.SiLU(inplace=False)
                )
            )

        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 16, 1),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim // 16, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 任务头
        self.heads = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.SiLU(inplace=False),
                nn.Linear(256, 1)
            ) for k in ("ph", "tem", "ogt", "tm")
        })
        self.to(self.device)

    def forward(self, x):
        # (B, L, 4096) → (B, hidden_dim)
        x = x.to(self.device, self.dtype).permute(0, 2, 1).contiguous()
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)        # 残差
        x = x * self.se(x)
        x = x.mean(dim=-1)        # 全局池化
        return [self.heads[k](x).squeeze(-1) for k in ("ph", "tem", "ogt", "tm")]

# ===================== UnifiedModel =====================

class UnifiedModel(nn.Module):
    def __init__(self, embed_module, forest_module, device="cuda"):
        super().__init__()
        self.embed_module = embed_module
        self.forest_module = forest_module
        self.device = torch.device(device)

        # 使用更稳定的参数初始化
        self.coef = nn.ParameterDict({
            'ph':  nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.02])),
            'tem': nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.02])),
            'ogt': nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.02])),
            'tm':  nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.02])),
        })
                
        # 任务上界
        self.scale = {'ph': 14.0, 'tem': 121.0, 'ogt': 100.0, 'tm': 50.0}

        # ---------- 多尺度 1-D CNN 主干 ----------
        self.conv1_3 = nn.Conv1d(4096, 1024, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(4096, 1024, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv1d(4096, 1024, kernel_size=7, padding=3)
        self.conv_reduce = nn.Conv1d(3 * 1024, 1024, kernel_size=1)

        self.res_block1 = ResBlock1D(1024)
        self.res_block2 = ResBlock1D(1024)

        # ---------- 轻量 MLP ----------
        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ---------- 各任务头 ----------
        self.mlp_ph  = MLP(1024, [512, 256], 1)
        self.mlp_tem = MLP(1024, [512, 256], 1)
        self.mlp_ogt = MLP(1024, [512, 256], 1)
        self.mlp_tm  = MLP(1024, [512, 256], 1)

        self.dtype = torch.float32
        self.ph_stagnation_counter = nn.Parameter(torch.tensor([0.5]))
        hidden_dim = 8
        self.fuse_mlp1 = nn.ModuleDict()
        self.fuse_mlp2 = nn.ModuleDict()
        self.residual_scale = nn.ParameterDict()

        for k in ['ph', 'tem', 'tm', 'ogt']:
            # 两层 MLP
            self.fuse_mlp1[k] = nn.Linear(3, hidden_dim)
            self.fuse_mlp2[k] = nn.Linear(hidden_dim, 1)
            # 残差权重 λ
            self.residual_scale[k] = nn.Parameter(torch.tensor(0.5))
            
        self.to(self.device)

    def _fuse_batch(self, e, f, m, key, flag):
        """批量融合处理: e, f, m 都是形状为[B]的张量"""
        weights = torch.softmax(self.coef[key][:4], dim=0)
        w_e, w_f, w_m, wef = weights
        
        # 基础融合 (向量化操作)
        base = (w_e * e + w_f * f + w_m * m + wef * e * f *m)

        if key == 'ph':
            # 1. 应用S形变换扩展范围
            y = 7.0 + 3.0 * torch.tanh((base - 7.0) / 3.0)  # 调整参数减少震荡
            
            # 2. 应用极端值增强
            acid_mask = y < 5.0
            base_mask = y > 9.0
            
            # 酸性增强：降低预测值
            y[acid_mask] = y[acid_mask] - 0.3 * torch.sigmoid(5.0 - y[acid_mask])
            
            # 碱性增强：提高预测值
            y[base_mask] = y[base_mask] + 0.3 * torch.sigmoid(y[base_mask] - 9.0)
            

            # 4. 确保值在合理范围
            y = torch.clamp(y, 0.0, 14.0)

        elif key == 'tm':
            # 保持原样 (向量化操作)
            y = torch.clamp(base, 0, 50)
        
        elif key == 'tem':
            # 极端温度增强 (向量化操作)
            y = base
            low_temp_boost = torch.sigmoid(50.0 - base)
            high_temp_boost = torch.sigmoid(base - 70.0)
            y = y * (1.0 + 0.2 * low_temp_boost + 0.2 * high_temp_boost)
            y = torch.clamp(y, 0, 121)
        
        elif key == 'ogt':
            # S形变换 (向量化操作)
            #centered = base - 50.0
            #y = 50.0 + 25.0 * torch.tanh(centered / 25.0)
            y=base
            y = torch.clamp(y, 0, 100)
        
        return y

    def forward(self, x, flag):
        # 输入x的形状: (B, L, 4096)
        x=x.unsqueeze(1)
        e_ph, e_tem, e_ogt, e_tm = self.embed_module(x)   # 每个都是[B]
        f_ph, f_tem, f_ogt, f_tm = self.forest_module(x)   # 每个都是[B]

        # ---------- 多尺度 CNN 主干 (批处理) ----------
        x_transposed = x.transpose(1, 2).to(self.dtype).contiguous()
        
        # 3 个分支 (批处理)
        b1 = F.gelu(self.conv1_3(x_transposed))
        b2 = F.gelu(self.conv1_5(x_transposed))
        b3 = F.gelu(self.conv1_7(x_transposed))
        x_cat = torch.cat([b1, b2, b3], dim=1)  # (B, 3*1024, L)
        
        # 减少通道数
        x_reduced = F.gelu(self.conv_reduce(x_cat))  # (B, 1024, L)
        
        # 残差块
        x_res = self.res_block1(x_reduced)
        x_res = self.res_block2(x_res)
        
        # 全局均值池化 -> (B, 1024)
        x_pooled = F.adaptive_avg_pool1d(x_res, 1).squeeze(-1)
        x_fc = self.fc(x_pooled)  # (B, 1024)

        # ---------- 各任务输出 (批处理) ----------
        m_ph = self.mlp_ph(x_fc).squeeze(-1)  # (B)
        m_tem = self.mlp_tem(x_fc).squeeze(-1)  # (B)
        m_ogt = self.mlp_ogt(x_fc).squeeze(-1)  # (B)
        m_tm = self.mlp_tm(x_fc).squeeze(-1)  # (B)

        # 融合预测结果 (批处理)
        ph = self._fuse_batch(e_ph, f_ph, m_ph, 'ph', flag)
        tem = self._fuse_batch(e_tem, f_tem, m_tem, 'tem', flag)
        ogt = self._fuse_batch(e_ogt, f_ogt, m_ogt, 'ogt', flag)
        tm = self._fuse_batch(e_tm, f_tm, m_tm, 'tm', flag)

        return ph, tem, ogt, tm

# ===================== 模型加载函数 =====================

def load_base_model_and_tokenizer(device='cuda'):
    """加载基础模型和tokenizer"""
    try:
        model_name = "Meta-Llama-3.1-8B-Instruct"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config
        )
        
        print(f"Base model loaded successfully, hidden size: {model.config.hidden_size}")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None, None

def load_hybrid_model(checkpoint_dir, task, device='cuda'):
    """加载混合模型"""
    try:
        model_path = os.path.join(checkpoint_dir, f"best_{task}_hybrid_model.pt")
        if not os.path.exists(model_path):
            print(f"Hybrid model not found: {model_path}")
            return None, None, None
        
        print(f"Loading hybrid model for {task} from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        input_dim = config.get('input_dim', 4096)
        hidden_dim = config.get('hidden_dim', 1024)
        alpha = checkpoint.get('alpha', 0.5)
        
        # 创建模型
        embed_module = SimpleProteinPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=0.1,
            device=device
        )
        
        forest_module = RandomForest(
            input_dim=input_dim,
            hidden_dim=256,
            num_blocks=4,
            dilations=(1, 2, 4, 8),
            device=device
        )
        
        model = UnifiedModel(embed_module, forest_module, device=device)
        model.load_state_dict(checkpoint['nn_model_state_dict'])
        model.eval()
        
        print(f"Hybrid model for {task} loaded successfully, alpha={alpha}")
        return model, alpha, checkpoint
        
    except Exception as e:
        print(f"Error loading hybrid model for {task}: {e}")
        return None, None, None

def load_ml_ensemble(checkpoint_dir, task):
    """加载ML模型集成"""
    try:
        ml_ensemble = MLModelEnsemble(task)
        if ml_ensemble.load(checkpoint_dir):
            print(f"ML ensemble for {task} loaded successfully")
            return ml_ensemble
        else:
            print(f"Failed to load ML ensemble for {task}")
            return None
    except Exception as e:
        print(f"Error loading ML ensemble for {task}: {e}")
        return None

# ===================== 预测函数 =====================

def predict_temperature(sequences, checkpoint_dir, tokenizer, base_model, device='cuda'):
    """预测温度"""
    print(f"Predicting temperature for {len(sequences)} sequences...")
    
    # 加载温度混合模型
    tem_model, tem_alpha, tem_checkpoint = load_hybrid_model(checkpoint_dir, 'tem', device)
    if tem_model is None:
        print("Failed to load temperature model")
        return []
    
    # 加载ML模型
    ml_ensemble_tem = load_ml_ensemble(checkpoint_dir, 'tem')
    
    # 创建缓存和特征提取器
    cache = SimpleSequenceCache(base_model, tokenizer, device)
    feature_extractor = ProteinFeatureExtractor()
    
    predictions = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # 神经网络预测
            embeds = []
            for seq in batch_sequences:
                embed = cache(seq)
                embeds.append(embed)
            
            if embeds:
                embeds = torch.cat(embeds, dim=0).to(device)
                _, tem_pred, _, _ = tem_model(embeds, flag=False)
                nn_predictions = tem_pred.cpu().numpy().flatten().tolist()
                
                # ML预测
                if ml_ensemble_tem:
                    ml_features = []
                    for seq in batch_sequences:
                        features = feature_extractor.extract_features(seq)
                        ml_features.append(features)
                    
                    ml_predictions = ml_ensemble_tem.predict(np.array(ml_features)).tolist()
                    
                    # 混合预测
                    for j in range(len(nn_predictions)):
                        hybrid_pred = tem_alpha * nn_predictions[j] + (1 - tem_alpha) * ml_predictions[j]
                        predictions.append(hybrid_pred)
                else:
                    predictions.extend(nn_predictions)
    
    return predictions

def predict_ph(sequences, checkpoint_dir, tokenizer, base_model, device='cuda'):
    """预测pH值"""
    print(f"Predicting pH for {len(sequences)} sequences...")
    
    # 加载pH混合模型
    ph_model, ph_alpha, ph_checkpoint = load_hybrid_model(checkpoint_dir, 'ph', device)
    if ph_model is None:
        print("Failed to load pH model")
        return []
    
    # 加载ML模型
    ml_ensemble_ph = load_ml_ensemble(checkpoint_dir, 'ph')
    
    # 创建缓存和特征提取器
    cache = SimpleSequenceCache(base_model, tokenizer, device)
    feature_extractor = ProteinFeatureExtractor()
    
    predictions = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # 神经网络预测
            embeds = []
            for seq in batch_sequences:
                embed = cache(seq)
                embeds.append(embed)
            
            if embeds:
                embeds = torch.cat(embeds, dim=0).to(device)
                ph_pred, _, _, _ = ph_model(embeds, flag=False)
                nn_predictions = ph_pred.cpu().numpy().flatten().tolist()
                
                # ML预测
                if ml_ensemble_ph:
                    ml_features = []
                    for seq in batch_sequences:
                        features = feature_extractor.extract_features(seq)
                        ml_features.append(features)
                    
                    ml_predictions = ml_ensemble_ph.predict(np.array(ml_features)).tolist()
                    
                    # 混合预测
                    for j in range(len(nn_predictions)):
                        hybrid_pred = ph_alpha * nn_predictions[j] + (1 - ph_alpha) * ml_predictions[j]
                        predictions.append(hybrid_pred)
                else:
                    predictions.extend(nn_predictions)
    
    return predictions

def read_csv_by_position(filename, sequence_col_index=1):
    """按列位置读取CSV文件，返回序列列表"""
    sequences = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行
            print(f"CSV header: {header}")
            
            for row_num, row in enumerate(reader, 2):
                if len(row) > sequence_col_index:
                    sequence = row[sequence_col_index].strip()
                    if sequence:  # 确保序列不为空
                        sequences.append(sequence)
                    else:
                        print(f"Warning: Empty sequence at row {row_num}")
                else:
                    print(f"Warning: Row {row_num} has insufficient columns")
        
        print(f"Loaded {len(sequences)} sequences from {filename}")
        return sequences
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

def read_ph_csv_by_position(filename):
    """专门读取pH CSV文件，按位置获取序列"""
    sequences = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # 尝试不同的分隔符
            try:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                print(f"pH CSV header (tab): {header}")
            except:
                f.seek(0)  # 回到文件开头
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                print(f"pH CSV header (comma): {header}")
            
            # 寻找序列列
            sequence_col_index = -1
            for i, col in enumerate(header):
                if 'sequence' in col.lower() or 'Sequence' in col:
                    sequence_col_index = i
                    break
            
            if sequence_col_index == -1:
                # 如果没有找到序列列，尝试按位置（第二列）
                sequence_col_index = 1
                print(f"Using column {sequence_col_index} as sequence column")
            
            f.seek(0)  # 回到文件开头
            next(reader)  # 跳过标题行
            
            for row_num, row in enumerate(reader, 2):
                if len(row) > sequence_col_index:
                    sequence = row[sequence_col_index].strip()
                    if sequence and len(sequence) > 10:  # 确保是有效的蛋白质序列
                        sequences.append(sequence)
                    else:
                        print(f"Warning: Invalid sequence at row {row_num}: '{sequence}'")
                else:
                    print(f"Warning: Row {row_num} has insufficient columns")
        
        print(f"Loaded {len(sequences)} pH sequences from {filename}")

        return sequences
    except Exception as e:
        print(f"Error reading pH file {filename}: {e}")
        return []

# ===================== 主函数 =====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict protein properties using hybrid models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_unified_all', 
                       help='Directory containing saved models')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use for inference (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载基础模型
    print("Loading base model and tokenizer...")
    tokenizer, base_model = load_base_model_and_tokenizer(device)
    if tokenizer is None:
        print("Failed to load base model")
        return
    
    # 冻结基础模型
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 读取温度数据并预测
    print("\n" + "="*80)
    print("PROCESSING TEMPERATURE DATA")
    print("="*80)
    
    # 按位置读取温度数据（第二列为序列）
    temperature_sequences = read_csv_by_position('../dataset/PHTem/tem/na_data_merged.csv', sequence_col_index=1)
    
    if not temperature_sequences:
        print("No temperature sequences found")
        return
    '''
    # 预测温度
    temperature_predictions = predict_temperature(temperature_sequences, args.checkpoint_dir, tokenizer, base_model, device)
    
    # 重新读取原始文件以获取完整数据
    temperature_data = []
    try:
        with open('../dataset/PHTem/tem/na_data_merged.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 4:
                    temperature_data.append({
                        'uniprot_id': row[0],
                        'sequence': row[1],
                        'ec_number': row[2],
                        'optimum_temperature': row[3]
                    })
    except Exception as e:
        print(f"Error re-reading temperature file: {e}")
        # 如果重新读取失败，使用序列列表创建基本数据
        temperature_data = [{'uniprot_id': f'SEQ_{i}', 'sequence': seq, 'ec_number': 'N/A', 'optimum_temperature': 'N/A'} 
                           for i, seq in enumerate(temperature_sequences)]
    
    # 写入温度预测结果
    output_temp_file = '../dataset/PHTem/tem/temperature_predictions.csv'
    with open(output_temp_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['UniProt ID', 'Sequence', 'EC Number', 'Predicted Temperature (°C)'])
        
        for i, item in enumerate(temperature_data):
            if i < len(temperature_predictions):
                predicted_temp = temperature_predictions[i]
                writer.writerow([
                    item['uniprot_id'],
                    item['sequence'],
                    item['ec_number'],
                    f"{predicted_temp:.2f}"
                ])
    
    print(f"Temperature predictions saved to: {output_temp_file}")
    '''
    # 读取pH数据并预测
    print("\n" + "="*80)
    print("PROCESSING PH DATA")
    print("="*80)
    
    # 使用专门函数读取pH数据
    ph_sequences = read_ph_csv_by_position('../dataset/PHTem/tem/enzyme_ph_all_invalid_unique.csv')
    
    if not ph_sequences:
        print("No pH sequences found")
        return
    
    # 预测pH
    #print(f"1111111111111111{len(ph_sequences)}")
    ph_predictions = predict_ph(ph_sequences, args.checkpoint_dir, tokenizer, base_model, device)
    
    # 重新读取原始文件以获取完整数据
    ph_data = []
    try:
        with open('../dataset/PHTem/tem/enzyme_ph_all_invalid_unique.csv', 'r', encoding='utf-8') as f:
            # 尝试不同的分隔符
            try:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
            except:
                f.seek(0)
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
            
            for row in reader:
                word=row[0].split(',')
                #print(word[1])
                if len(word) >= 3:  # 至少需要3列
                    ph_data.append({
                        'uniprot_id': word[0],
                        'sequence': word[2] ,
                        'ec_number': word[1], 
                        'optimum_ph': word[3],
                    })
        #print(f"1111111111111111{len(word)}")
    except Exception as e:
        print(f"Error re-reading pH file: {e}")
        # 如果重新读取失败，使用序列列表创建基本数据
        ph_data = [{'uniprot_id': f'SEQ_{i}', 'sequence': seq, 'ec_number': 'N/A', 'optimum_ph': 'N/A'} 
                  for i, seq in enumerate(ph_sequences)]
    
    # 写入pH预测结果
    output_ph_file = '../dataset/PHTem/tem/ph_predictions.csv'
    with open(output_ph_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['UniProt ID', 'Sequence', 'EC Number', 'Predicted pH'])
        
        for i, item in enumerate(ph_data):
            if i < len(ph_predictions):
                predicted_ph = ph_predictions[i]
                writer.writerow([
                    item['uniprot_id'],
                    item['sequence'],
                    item['ec_number'],
                    f"{predicted_ph:.2f}"
                ])
    
    print(f"pH predictions saved to: {output_ph_file}")
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Temperature predictions: {output_temp_file}")
    print(f"pH predictions: {output_ph_file}")

if __name__ == '__main__':
    main()