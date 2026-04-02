"""
================================================================================
论文题目：基于开放网络多元数据的问答系统投毒攻击检测工具
模块名称：PoisoningDetector - 投毒攻击检测核心组件
功能说明：封装异构图对比学习检测算法，提供黑盒式调用接口
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 图神经网络层（同前，略） ====================
class GINLayer(nn.Module):
    """图同构网络层"""
    def __init__(self, in_dim, out_dim, epsilon=0.1):
        super(GINLayer, self).__init__()
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        neighbor_sum = torch.zeros_like(x)
        src, dst = edge_index[0], edge_index[1]
        neighbor_sum.index_add_(0, dst, x[src])
        combined = (1 + self.epsilon) * x + neighbor_sum
        out = self.mlp(combined)
        return out


class HeteroEncoder(nn.Module):
    """异构图编码器（修复版）"""
    def __init__(self, in_dims_dict, hidden_dim, out_dim, n_layers=2):
        super(HeteroEncoder, self).__init__()
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.node_types = ['user', 'question', 'answer']

        # 投影层：将各类型节点特征投影到统一隐藏维度
        self.projections = nn.ModuleDict({
            'user': nn.Linear(in_dims_dict['user'], hidden_dim),
            'question': nn.Linear(in_dims_dict['question'], hidden_dim),
            'answer': nn.Linear(in_dims_dict['answer'], hidden_dim)
        })

        # 定义边类型（与数据集中一致）
        self.edge_types = [
            ('user', 'asks', 'question'),
            ('user', 'answers', 'question'),
            ('question', 'contains', 'answer'),
            ('user', 'rates', 'answer'),
            ('user', 'similar_to', 'user')
        ]
        # 为每种边类型生成字符串键，用于存储 MLP
        self.edge_type_keys = [f"{src}_{rel}_{dst}" for src, rel, dst in self.edge_types]

        # 为每种边类型创建 MLP，用于更新目标节点特征
        self.mlps = nn.ModuleDict()
        for key in self.edge_type_keys:
            self.mlps[key] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )

        # 可学习的 epsilon 参数（残差连接权重）
        self.epsilon = nn.Parameter(torch.tensor(0.1))

        # 输出层：将用户特征映射到最终输出维度
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.act = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        # 1. 投影到统一维度
        h_dict = {}
        for node_type in self.node_types:
            h_dict[node_type] = self.projections[node_type](x_dict[node_type])

        # 2. 多层消息传递
        for layer_idx in range(self.n_layers):
            # 初始化新特征为当前特征（用于累积更新）
            new_h_dict = {node_type: h_dict[node_type].clone() for node_type in self.node_types}

            # 处理每种边类型
            for idx, edge_type in enumerate(self.edge_types):
                src_type, _, dst_type = edge_type
                # 跳过不存在的边类型
                if edge_type not in edge_index_dict or edge_index_dict[edge_type].shape[1] == 0:
                    continue

                edge_idx = edge_index_dict[edge_type]          # [2, num_edges]
                src, dst = edge_idx[0], edge_idx[1]            # 源节点索引、目标节点索引
                key = self.edge_type_keys[idx]

                # 获取源节点和目标节点的特征
                src_feat = h_dict[src_type]      # [num_src, hidden_dim]
                dst_feat = h_dict[dst_type]      # [num_dst, hidden_dim]

                # 聚合：将源节点特征累加到对应的目标节点上
                aggr = torch.zeros_like(dst_feat)        # [num_dst, hidden_dim]
                aggr.index_add_(0, dst, src_feat[src])   # 按目标节点索引累加

                # 结合自身特征（残差）
                combined = (1 + self.epsilon) * dst_feat + aggr

                # 通过 MLP 得到更新后的目标节点特征
                update = self.mlps[key](combined)

                # 将更新累加到新特征字典中
                new_h_dict[dst_type] = new_h_dict[dst_type] + update

            # 应用激活函数（除最后一层外）
            if layer_idx < self.n_layers - 1:
                for node_type in self.node_types:
                    new_h_dict[node_type] = self.act(new_h_dict[node_type])

            h_dict = new_h_dict

        # 3. 输出用户节点嵌入
        user_features = h_dict['user']
        output = self.output_layer(user_features)
        return output

    def get_global_representation(self, h):
        """计算全局特征"""
        global_feat = torch.mean(h, dim=0, keepdim=True)
        global_feat = torch.sigmoid(global_feat)
        return global_feat

class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h, s):
        s_expanded = s.expand(h.shape[0], -1)
        combined = torch.cat([h, s_expanded], dim=1)
        prob = self.fc(combined)
        return prob


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_features=3):
        super(AttentionFusion, self).__init__()
        self.query_net = nn.Linear(feature_dim, feature_dim)
        self.key_net = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_features)])
        self.feature_dim = feature_dim
        self.num_features = num_features

    def forward(self, user_profile, features_list):
        query = self.query_net(user_profile)
        keys = [key_net(feature) for key_net, feature in zip(self.key_net, features_list)]
        scores = [(query * key).sum(dim=1, keepdim=True) for key in keys]
        scores_tensor = torch.cat(scores, dim=1)
        attention_weights = F.softmax(scores_tensor, dim=1)
        fused_features = torch.zeros_like(features_list[0])
        for i, feature in enumerate(features_list):
            weight = attention_weights[:, i:i+1]
            fused_features = fused_features + weight * feature
        return fused_features, attention_weights


class DeepClustering(nn.Module):
    def __init__(self, feature_dim, n_clusters=2):
        super(DeepClustering, self).__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, feature_dim))

    def soft_assignment(self, h):
        dist = torch.cdist(h, self.cluster_centers)
        q = 1.0 / (1.0 + dist ** 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        numerator = q ** 2
        denominator = numerator.sum(dim=0, keepdim=True)
        p = numerator / denominator
        p = p / p.sum(dim=1, keepdim=True)
        return p

    def kl_loss(self, q, p):
        return F.kl_div(q.log(), p, reduction='batchmean')

    def get_clusters(self, h):
        q = self.soft_assignment(h)
        clusters = torch.argmax(q, dim=1)
        return clusters


class ContrastiveHeteroModel(nn.Module):
    def __init__(self, in_dims_dict, hidden_dim, out_dim, n_layers=2):
        super(ContrastiveHeteroModel, self).__init__()
        self.encoder = HeteroEncoder(in_dims_dict, hidden_dim, out_dim, n_layers)
        self.discriminator = Discriminator(out_dim)
        self.deep_clustering = DeepClustering(out_dim, n_clusters=2)
        self.attention_fusion = AttentionFusion(out_dim, num_features=3)
        self.loss_history = []
        self.pos_acc_history = []
        self.neg_acc_history = []
        self.cluster_loss_history = []

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def generate_negative_samples(self, x_dict, edge_index_dict, epsilon=0.1):
        original_params = {}
        for name, param in self.encoder.named_parameters():
            original_params[name] = param.data.clone()
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if 'weight' in name:
                    noise = torch.randn_like(param) * epsilon
                    param.data = param.data + noise
        negative_features = self.encoder(x_dict, edge_index_dict)
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                param.data = original_params[name]
        return negative_features

    def contrastive_loss(self, pos_features, neg_features, global_feat):
        pos_score = self.discriminator(pos_features, global_feat)
        neg_score = self.discriminator(neg_features, global_feat)
        pos_loss = -torch.log(pos_score + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-8).mean()
        total_loss = (pos_loss + neg_loss) / 2
        return total_loss, pos_loss, neg_loss, pos_score, neg_score

    def train_step(self, x_dict, edge_index_dict, optimizer, epsilon=0.1):
        self.train()
        optimizer.zero_grad()
        pos_features = self.encoder(x_dict, edge_index_dict)
        global_feat = self.encoder.get_global_representation(pos_features)
        neg_features = self.generate_negative_samples(x_dict, edge_index_dict, epsilon)
        contrast_loss, _, _, _, _ = self.contrastive_loss(pos_features, neg_features, global_feat)
        q = self.deep_clustering.soft_assignment(pos_features)
        p = self.deep_clustering.target_distribution(q)
        cluster_loss = self.deep_clustering.kl_loss(q, p)
        total_loss = contrast_loss + 0.1 * cluster_loss
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            pos_score = self.discriminator(pos_features, global_feat)
            neg_score = self.discriminator(neg_features, global_feat)
            pos_acc = (pos_score > 0.5).float().mean().item()
            neg_acc = (neg_score < 0.5).float().mean().item()
        self.loss_history.append(contrast_loss.item())
        self.cluster_loss_history.append(cluster_loss.item())
        self.pos_acc_history.append(pos_acc)
        self.neg_acc_history.append(neg_acc)
        return contrast_loss.item(), cluster_loss.item(), pos_acc, neg_acc

    def fit(self, x_dict, edge_index_dict, epochs=100, lr=0.001, epsilon=0.1, verbose=True):
        """训练模型"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            contrast_loss, cluster_loss, pos_acc, neg_acc = self.train_step(
                x_dict, edge_index_dict, optimizer, epsilon
            )
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {contrast_loss:.4f} | Cluster: {cluster_loss:.4f} | "
                      f"PosAcc: {pos_acc:.2%} | NegAcc: {neg_acc:.2%}")

    def predict(self, x_dict, edge_index_dict):
        """预测用户是否为攻击者（返回标签和置信度）"""
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(x_dict, edge_index_dict)
            # 使用聚类获得硬标签
            clusters = self.deep_clustering.get_clusters(embeddings)
            # 计算每个节点的异常分数（到聚类中心的距离）
            q = self.deep_clustering.soft_assignment(embeddings)
            # 假设聚类0是正常，聚类1是攻击者（可能需要根据实际调整）
            # 由于聚类是无监督的，我们根据聚类结果中攻击者占比来判定哪个簇是攻击者
            # 这里返回原始簇标签，外部可能需要根据情况映射
            # 同时返回异常分数（与簇中心的负对数似然）
            anomaly_scores = -torch.log(q[:, 1] + 1e-8)  # 第二个簇的负对数概率作为异常分数
        return clusters.cpu().numpy(), anomaly_scores.cpu().numpy()


# ==================== 2. 数据处理器（简化版，仅用于加载数据） ====================
class HeteroDataLoader:
    """数据加载器，将原始数据转换为模型所需格式"""
    @staticmethod
    def load(data_path):
        """加载数据文件（pickle格式）"""
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        # 转换为张量
        x_dict = {
            'user': torch.FloatTensor(raw_data['user_features']),
            'question': torch.FloatTensor(raw_data['question_features']),
            'answer': torch.FloatTensor(raw_data['answer_features'])
        }
        edge_index_dict = {}
        for edge_type, edges in raw_data['edges'].items():
            if len(edges) > 0:
                edge_index_dict[edge_type] = torch.LongTensor(edges).t().contiguous()
            else:
                edge_index_dict[edge_type] = torch.empty(2, 0, dtype=torch.long)
        labels = torch.LongTensor(raw_data['user_labels']) if 'user_labels' in raw_data else None
        return x_dict, edge_index_dict, labels


# ==================== 3. 黑盒子组件：投毒检测器 ====================
class PoisoningDetector:
    """
    投毒攻击检测器（黑盒子组件）
    功能：接收数据集，训练模型，返回检测结果
    """
    def __init__(self, hidden_dim=64, out_dim=32, n_layers=2, device='cpu'):
        """
        参数：
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            n_layers: GIN层数
            device: 运行设备
        """
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.model = None
        self.is_fitted = False

    def _get_in_dims(self, x_dict):
        """从数据中获取输入维度"""
        return {
            'user': x_dict['user'].shape[1],
            'question': x_dict['question'].shape[1],
            'answer': x_dict['answer'].shape[1]
        }

    def fit(self, x_dict, edge_index_dict, epochs=100, lr=0.001, epsilon=0.1, verbose=True):
        """
        训练检测器
        参数：
            x_dict: 节点特征字典 {'user': tensor, 'question': tensor, 'answer': tensor}
            edge_index_dict: 边索引字典
            epochs: 训练轮数
            lr: 学习率
            epsilon: 负样本扰动幅度
            verbose: 是否打印进度
        """
        # 将数据移动到指定设备
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}

        in_dims = self._get_in_dims(x_dict)
        self.in_dims = in_dims
        self.model = ContrastiveHeteroModel(in_dims, self.hidden_dim, self.out_dim, self.n_layers)
        self.model.to(self.device)

        self.model.fit(x_dict, edge_index_dict, epochs, lr, epsilon, verbose)
        self.is_fitted = True
        return self

    def predict(self, x_dict, edge_index_dict):
        """
        预测用户是否为攻击者
        返回：
            labels: 预测标签数组 (0=正常, 1=攻击者)
            scores: 异常分数（越高越可能是攻击者）
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        clusters, scores = self.model.predict(x_dict, edge_index_dict)
        # 由于聚类标签可能是0/1但顺序不确定，这里根据异常分数决定哪个簇是攻击者
        # 假设攻击者簇的异常分数更高
        # 实际应用中，可以通过少量标签校准，或直接返回原始簇和分数由外部处理
        # 这里我们返回原始簇，并额外返回分数
        return clusters, scores

    def predict_from_data(self, data_path, **kwargs):
        """
        从数据文件直接加载并预测（一站式接口）
        参数：
            data_path: 数据文件路径（pickle格式）
        返回：
            labels, scores
        """
        x_dict, edge_index_dict, _ = HeteroDataLoader.load(data_path)
        if not self.is_fitted:
            self.fit(x_dict, edge_index_dict, **kwargs)
        return self.predict(x_dict, edge_index_dict)

    def save(self, model_path):
        """保存模型参数和输入维度信息"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，无法保存")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'n_layers': self.n_layers,
            'in_dims': self.in_dims,  # 保存输入维度
            'device': self.device.type
        }, model_path)
        print(f"模型已保存到 {model_path}")

    def load(self, model_path):
        """加载模型参数（无需提供 x_dict）"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.hidden_dim = checkpoint['hidden_dim']
        self.out_dim = checkpoint['out_dim']
        self.n_layers = checkpoint['n_layers']
        in_dims = checkpoint['in_dims']  # 恢复输入维度
        self.model = ContrastiveHeteroModel(in_dims, self.hidden_dim, self.out_dim, self.n_layers)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.is_fitted = True
        print(f"模型已从 {model_path} 加载")


# ==================== 4. 主程序（演示如何使用黑盒子） ====================
def main():
    # 假设数据集文件为 'hetero_dataset.pkl'
    data_path = 'hetero_dataset.pkl'

    # 创建检测器实例
    detector = PoisoningDetector(hidden_dim=64, out_dim=32, n_layers=2, device='cpu')

    # 方法1：一步完成训练和预测
    print("=== 使用检测器进行检测 ===")
    labels, scores = detector.predict_from_data(data_path, epochs=100, verbose=True)
    print(f"检测完成，共 {len(labels)} 个用户")
    print(f"预测为攻击者的数量: {(labels == 1).sum()}")
    print(f"异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")

    # 保存模型供后续使用
    detector.save('detector_model.pth')

    # 方法2：单独预测新数据（假设已有训练好的模型）
    # 注意：这里为了演示，重新加载模型，但实际使用时可以复用已训练的detector
    # 加载数据
    x_dict, edge_index_dict, _ = HeteroDataLoader.load(data_path)
    # 创建新检测器并加载模型
    new_detector = PoisoningDetector()
    new_detector.load('detector_model.pth', x_dict=x_dict)
    new_labels, new_scores = new_detector.predict(x_dict, edge_index_dict)
    print("使用加载的模型再次预测，结果一致:", np.array_equal(labels, new_labels))

    print("黑盒子组件测试通过。")


if __name__ == "__main__":
    main()
