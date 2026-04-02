"""
================================================================================
数据集生成模块：面向问答系统的异构图数据集
模块名称：data_c.py
功能说明：生成用于异构图对比学习检测算法的模拟数据集
         包含用户、问题、答案三种节点，以及五种边类型
         同时模拟正常用户和攻击者（水军）的行为模式
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import pickle

# ==================== 设置随机种子 ====================
np.random.seed(42)
random.seed(42)


# ==================== 1. 配置参数 ====================
class DatasetConfig:
    """数据集配置参数"""
    # 基础数量
    N_USERS = 1000  # 总用户数
    N_QUESTIONS = 500  # 问题数
    N_ANSWERS = 800  # 答案数
    N_ATTACKERS = 80  # 攻击者数量（占总用户8%）

    # 攻击者比例
    ATTACKER_RATIO = 0.08  # 攻击者比例

    # 特征维度
    USER_FEATURE_DIM = 8  # 用户行为特征维度
    QUESTION_FEATURE_DIM = 4  # 问题特征维度
    ANSWER_FEATURE_DIM = 4  # 答案特征维度

    # 边数量控制
    N_ASK_EDGES = 800  # 提问边数量
    N_ANSWER_EDGES = 1200  # 回答边数量
    N_CONTAINS_EDGES = 800  # 问题包含答案边数量
    N_RATE_EDGES = 2000  # 评分边数量
    N_SIMILAR_EDGES = 500  # 用户相似边数量


class DataGenerator:
    """
    数据生成器
    功能：生成模拟的问答系统数据，包含正常用户和攻击者
    """

    def __init__(self, config=DatasetConfig()):
        self.config = config
        self.data = {}

    def generate_user_ids(self):
        """生成用户ID列表"""
        return list(range(self.config.N_USERS))

    def generate_question_ids(self):
        """生成问题ID列表"""
        return list(range(self.config.N_QUESTIONS))

    def generate_answer_ids(self):
        """生成答案ID列表"""
        return list(range(self.config.N_ANSWERS))

    # ==================== 2. 生成用户行为特征 ====================
    def generate_user_features(self, attacker_indices):
        """
        生成用户行为特征（8维）

        特征说明：
        feat[0]: 提问次数（归一化）           - 正常用户：0-0.3，攻击者：0.5-1.0
        feat[1]: 回答次数（归一化）           - 正常用户：0-0.4，攻击者：0.6-1.0
        feat[2]: 评分次数（归一化）           - 正常用户：0-0.5，攻击者：0.7-1.0
        feat[3]: 平均评分值（归一化）         - 正常用户：0.4-0.8，攻击者：0或1（极端）
        feat[4]: 评分标准差（归一化）         - 正常用户：0.1-0.4，攻击者：0-0.1（过于稳定）
        feat[5]: 夜间活动比例（0-1）          - 正常用户：0-0.2，攻击者：0.6-1.0
        feat[6]: 平均回答长度（归一化）       - 正常用户：0.3-0.8，攻击者：0-0.2（短回答）
        feat[7]: 行为突发性指数（0-1）        - 正常用户：0-0.4，攻击者：0.7-1.0
        """
        n_users = self.config.N_USERS
        n_attackers = len(attacker_indices)

        # 初始化特征矩阵
        user_features = np.zeros((n_users, self.config.USER_FEATURE_DIM))

        # 正常用户特征（正态分布，集中在正常范围）
        normal_count = n_users - n_attackers
        normal_indices = [i for i in range(n_users) if i not in attacker_indices]

        # 正常用户：提问次数（少量）
        user_features[normal_indices, 0] = np.random.beta(2, 5, normal_count) * 0.3

        # 正常用户：回答次数（中等）
        user_features[normal_indices, 1] = np.random.beta(3, 4, normal_count) * 0.4

        # 正常用户：评分次数（中等）
        user_features[normal_indices, 2] = np.random.beta(3, 3, normal_count) * 0.5

        # 正常用户：平均评分（3-4分左右）
        user_features[normal_indices, 3] = np.random.beta(4, 2, normal_count) * 0.6 + 0.2

        # 正常用户：评分标准差（有一定波动）
        user_features[normal_indices, 4] = np.random.beta(2, 4, normal_count) * 0.5

        # 正常用户：夜间活动比例（白天为主）
        user_features[normal_indices, 5] = np.random.beta(1, 8, normal_count)

        # 正常用户：平均回答长度（中等偏长）
        user_features[normal_indices, 6] = np.random.beta(5, 3, normal_count) * 0.7 + 0.2

        # 正常用户：突发性指数（平稳）
        user_features[normal_indices, 7] = np.random.beta(2, 6, normal_count) * 0.4

        # 攻击者特征（异常分布）
        # 攻击者：提问次数高（刷屏）
        user_features[attacker_indices, 0] = np.random.beta(5, 2, n_attackers) * 0.5 + 0.5

        # 攻击者：回答次数高
        user_features[attacker_indices, 1] = np.random.beta(6, 2, n_attackers) * 0.4 + 0.6

        # 攻击者：评分次数高
        user_features[attacker_indices, 2] = np.random.beta(7, 2, n_attackers) * 0.3 + 0.7

        # 攻击者：平均评分极端（要么全5分，要么全1分）
        extreme = np.random.choice([0, 1], n_attackers, p=[0.5, 0.5])
        user_features[attacker_indices, 3] = extreme * 1.0  # 极端值

        # 攻击者：评分标准差极小（评分一致）
        user_features[attacker_indices, 4] = np.random.beta(1, 10, n_attackers) * 0.1

        # 攻击者：夜间活动比例高
        user_features[attacker_indices, 5] = np.random.beta(8, 2, n_attackers) * 0.5 + 0.5

        # 攻击者：平均回答长度短（如"好好好"）
        user_features[attacker_indices, 6] = np.random.beta(1, 8, n_attackers) * 0.2

        # 攻击者：突发性指数高
        user_features[attacker_indices, 7] = np.random.beta(8, 2, n_attackers) * 0.5 + 0.5

        # 添加少量噪声
        user_features += np.random.randn(n_users, self.config.USER_FEATURE_DIM) * 0.05
        user_features = np.clip(user_features, 0, 1)  # 归一化到[0,1]

        return user_features.astype(np.float32)

    # ==================== 3. 生成问题特征 ====================
    def generate_question_features(self):
        """
        生成问题特征（4维）

        特征说明：
        feat[0]: 标题长度（归一化）
        feat[1]: 是否包含问号（0/1）
        feat[2]: 发布时间（是否夜间）
        feat[3]: 浏览数（归一化）
        """
        n_questions = self.config.N_QUESTIONS

        question_features = np.zeros((n_questions, self.config.QUESTION_FEATURE_DIM))

        # 标题长度：大部分适中，少数很长
        question_features[:, 0] = np.random.beta(3, 3, n_questions)

        # 是否包含问号：大部分有问题含问号
        question_features[:, 1] = np.random.binomial(1, 0.8, n_questions)

        # 发布时间：大部分白天
        question_features[:, 2] = np.random.beta(1, 7, n_questions)

        # 浏览数：长尾分布
        question_features[:, 3] = np.random.pareto(2, n_questions)
        question_features[:, 3] = np.clip(question_features[:, 3] / question_features[:, 3].max(), 0, 1)

        return question_features.astype(np.float32)

    # ==================== 4. 生成答案特征 ====================
    def generate_answer_features(self):
        """
        生成答案特征（4维）

        特征说明：
        feat[0]: 内容长度（归一化）- 攻击者的答案通常很短
        feat[1]: 是否夜间发布
        feat[2]: 是否极短回答（<20字）
        feat[3]: 点赞数（归一化）
        """
        n_answers = self.config.N_ANSWERS

        answer_features = np.zeros((n_answers, self.config.ANSWER_FEATURE_DIM))

        # 内容长度：正态分布
        answer_features[:, 0] = np.random.beta(4, 3, n_answers)

        # 是否夜间发布
        answer_features[:, 1] = np.random.beta(1, 6, n_answers)

        # 是否极短回答
        answer_features[:, 2] = np.random.binomial(1, 0.1, n_answers)

        # 点赞数
        answer_features[:, 3] = np.random.pareto(2, n_answers)
        answer_features[:, 3] = np.clip(answer_features[:, 3] / answer_features[:, 3].max(), 0, 1)

        return answer_features.astype(np.float32)

    # ==================== 5. 生成边：用户-问题（提问） ====================
    def generate_ask_edges(self, attacker_indices):
        """
        生成提问边（用户 -> 问题）
        攻击者会提问更多问题
        """
        edges = []
        n_edges = self.config.N_ASK_EDGES

        # 攻击者的提问概率更高
        attacker_prob = 0.6  # 攻击者占提问边的60%
        normal_prob = 0.4  # 正常用户占40%

        n_attacker_edges = int(n_edges * attacker_prob)
        n_normal_edges = n_edges - n_attacker_edges

        # 攻击者提问
        for _ in range(n_attacker_edges):
            user = np.random.choice(attacker_indices)
            question = np.random.randint(0, self.config.N_QUESTIONS)
            edges.append([user, question])

        # 正常用户提问
        normal_users = [i for i in range(self.config.N_USERS) if i not in attacker_indices]
        for _ in range(n_normal_edges):
            user = np.random.choice(normal_users)
            question = np.random.randint(0, self.config.N_QUESTIONS)
            edges.append([user, question])

        return np.array(edges)

    # ==================== 6. 生成边：用户-问题（回答） ====================
    def generate_answer_edges(self, attacker_indices):
        """
        生成回答边（用户 -> 问题）
        攻击者会回答更多问题（刷屏）
        """
        edges = []
        n_edges = self.config.N_ANSWER_EDGES

        attacker_prob = 0.65  # 攻击者占65%
        normal_prob = 0.35

        n_attacker_edges = int(n_edges * attacker_prob)
        n_normal_edges = n_edges - n_attacker_edges

        # 攻击者回答
        for _ in range(n_attacker_edges):
            user = np.random.choice(attacker_indices)
            question = np.random.randint(0, self.config.N_QUESTIONS)
            edges.append([user, question])

        # 正常用户回答
        normal_users = [i for i in range(self.config.N_USERS) if i not in attacker_indices]
        for _ in range(n_normal_edges):
            user = np.random.choice(normal_users)
            question = np.random.randint(0, self.config.N_QUESTIONS)
            edges.append([user, question])

        return np.array(edges)

    # ==================== 7. 生成边：问题-答案（包含） ====================
    def generate_contains_edges(self, attacker_answers=None):
        """
        生成问题包含答案边（问题 -> 答案）
        """
        edges = []
        n_edges = self.config.N_CONTAINS_EDGES

        # 每个答案只能属于一个问题
        used_answers = set()

        for _ in range(n_edges):
            question = np.random.randint(0, self.config.N_QUESTIONS)
            answer = np.random.randint(0, self.config.N_ANSWERS)

            # 避免重复
            while answer in used_answers:
                answer = np.random.randint(0, self.config.N_ANSWERS)
            used_answers.add(answer)

            edges.append([question, answer])

        return np.array(edges)

    # ==================== 8. 生成边：用户-答案（评分） ====================
    def generate_rate_edges(self, attacker_indices):
        """
        生成评分边（用户 -> 答案）
        攻击者的评分模式极端（全5分或全1分）

        返回：边索引和评分值
        """
        edges = []
        ratings = []
        n_edges = self.config.N_RATE_EDGES

        # 攻击者比例
        attacker_prob = 0.5
        n_attacker_edges = int(n_edges * attacker_prob)
        n_normal_edges = n_edges - n_attacker_edges

        normal_users = [i for i in range(self.config.N_USERS) if i not in attacker_indices]

        # 攻击者评分（极端值）
        for _ in range(n_attacker_edges):
            user = np.random.choice(attacker_indices)
            answer = np.random.randint(0, self.config.N_ANSWERS)
            edges.append([user, answer])

            # 随机选择极端评分（1分或5分）
            extreme_score = np.random.choice([1, 5], p=[0.3, 0.7])
            ratings.append(extreme_score)

        # 正常用户评分（正态分布）
        for _ in range(n_normal_edges):
            user = np.random.choice(normal_users)
            answer = np.random.randint(0, self.config.N_ANSWERS)
            edges.append([user, answer])

            # 正常评分（2-5分）
            normal_score = np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
            ratings.append(normal_score)

        edges_array = np.array(edges)
        ratings_array = np.array(ratings)

        return edges_array, ratings_array

    # ==================== 9. 生成边：用户-用户（相似度） ====================
    def generate_similar_edges(self, attacker_indices):
        """
        生成用户相似边（用户 -> 用户）
        攻击者之间会形成紧密的小团体（互相连接）
        """
        edges = []
        n_edges = self.config.N_SIMILAR_EDGES

        # 攻击者之间的连接（团伙作弊）
        n_attacker_edges = int(n_edges * 0.7)

        # 攻击者之间互相连接
        for _ in range(n_attacker_edges):
            user1 = np.random.choice(attacker_indices)
            user2 = np.random.choice(attacker_indices)
            if user1 != user2:
                edges.append([user1, user2])

        # 正常用户之间的连接（随机）
        normal_users = [i for i in range(self.config.N_USERS) if i not in attacker_indices]
        n_normal_edges = n_edges - len(edges)

        for _ in range(n_normal_edges):
            if len(normal_users) >= 2:
                user1 = np.random.choice(normal_users)
                user2 = np.random.choice(normal_users)
                if user1 != user2:
                    edges.append([user1, user2])

        return np.array(edges)

    # ==================== 10. 生成攻击者团伙 ====================
    def generate_attacker_groups(self):
        """
        生成攻击者团伙（协同作弊）
        攻击者之间会形成多个小团体
        """
        n_attackers = self.config.N_ATTACKERS
        attacker_indices = list(range(self.config.N_USERS - n_attackers, self.config.N_USERS))

        # 将攻击者分成几个小团体
        n_groups = 4
        group_size = n_attackers // n_groups

        groups = []
        for i in range(n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < n_groups - 1 else n_attackers
            groups.append(attacker_indices[start:end])

        # 为每个团体添加内部高相似度
        # 这里在边生成时会体现

        return attacker_indices, groups

    # ==================== 11. 生成完整数据集 ====================
    def generate_dataset(self):
        """
        生成完整的数据集
        """
        print("=" * 60)
        print("开始生成问答系统异构图数据集...")
        print("=" * 60)

        # 1. 生成攻击者
        n_attackers = self.config.N_ATTACKERS
        attacker_indices = list(range(self.config.N_USERS - n_attackers, self.config.N_USERS))
        print(f"✓ 攻击者数量: {n_attackers} (用户ID {attacker_indices[0]}~{attacker_indices[-1]})")

        # 2. 生成节点特征
        print("\n生成节点特征...")
        user_features = self.generate_user_features(attacker_indices)
        print(f"  用户特征: {user_features.shape}")

        question_features = self.generate_question_features()
        print(f"  问题特征: {question_features.shape}")

        answer_features = self.generate_answer_features()
        print(f"  答案特征: {answer_features.shape}")

        # 3. 生成边
        print("\n生成边关系...")

        ask_edges = self.generate_ask_edges(attacker_indices)
        print(f"  提问边: {ask_edges.shape[0]} 条")

        answer_edges = self.generate_answer_edges(attacker_indices)
        print(f"  回答边: {answer_edges.shape[0]} 条")

        contains_edges = self.generate_contains_edges()
        print(f"  包含边: {contains_edges.shape[0]} 条")

        rate_edges, rating_values = self.generate_rate_edges(attacker_indices)
        print(f"  评分边: {rate_edges.shape[0]} 条")

        similar_edges = self.generate_similar_edges(attacker_indices)
        print(f"  相似边: {similar_edges.shape[0]} 条")

        # 4. 生成标签（用户是否为攻击者）
        user_labels = np.zeros(self.config.N_USERS, dtype=np.int64)
        user_labels[attacker_indices] = 1

        normal_count = (user_labels == 0).sum()
        attacker_count = (user_labels == 1).sum()
        print(f"\n用户标签分布:")
        print(f"  正常用户: {normal_count}")
        print(f"  攻击者: {attacker_count}")

        # 5. 组织边字典
        edges_dict = {
            ('user', 'asks', 'question'): ask_edges,
            ('user', 'answers', 'question'): answer_edges,
            ('question', 'contains', 'answer'): contains_edges,
            ('user', 'rates', 'answer'): rate_edges,
            ('user', 'similar_to', 'user'): similar_edges
        }

        # 6. 组织评分值字典
        rating_attrs_dict = {
            ('user', 'rates', 'answer'): rating_values
        }

        # 7. 组装数据
        dataset = {
            'user_features': user_features,
            'question_features': question_features,
            'answer_features': answer_features,
            'edges': edges_dict,
            'rating_attrs': rating_attrs_dict,
            'user_labels': user_labels,
            'config': {
                'n_users': self.config.N_USERS,
                'n_questions': self.config.N_QUESTIONS,
                'n_answers': self.config.N_ANSWERS,
                'n_attackers': n_attackers,
                'attacker_indices': attacker_indices,
                'user_feature_dim': self.config.USER_FEATURE_DIM,
                'question_feature_dim': self.config.QUESTION_FEATURE_DIM,
                'answer_feature_dim': self.config.ANSWER_FEATURE_DIM
            }
        }

        return dataset

    # ==================== 12. 保存数据集 ====================
    def save_dataset(self, dataset, filename='hetero_dataset.npz'):
        """
        保存数据集为npz文件
        """
        # 准备保存的数据（需要序列化边字典）
        save_data = {
            'user_features': dataset['user_features'],
            'question_features': dataset['question_features'],
            'answer_features': dataset['answer_features'],
            'edges': dataset['edges'],  # 字典需要特殊处理
            'rating_attrs': dataset['rating_attrs'],
            'user_labels': dataset['user_labels'],
            'config': str(dataset['config'])
        }

        # 使用pickle保存包含字典的数据
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\n✓ 数据集已保存到: {filename}")

        return filename

    # ==================== 13. 加载数据集 ====================
    @staticmethod
    def load_dataset(filename='hetero_dataset.npz'):
        """
        加载数据集
        """
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)

        print(f"✓ 数据集已从 {filename} 加载")
        return dataset

    # ==================== 14. 数据集统计信息 ====================
    def print_statistics(self, dataset):
        """
        打印数据集统计信息
        """
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)

        # 节点统计
        print("\n【节点统计】")
        print(f"  用户节点: {dataset['user_features'].shape[0]}")
        print(f"  问题节点: {dataset['question_features'].shape[0]}")
        print(f"  答案节点: {dataset['answer_features'].shape[0]}")

        # 边统计
        print("\n【边统计】")
        edges = dataset['edges']
        for edge_type, edge_array in edges.items():
            print(f"  {edge_type}: {edge_array.shape[0]} 条")

        # 标签统计
        print("\n【标签统计】")
        labels = dataset['user_labels']
        normal = (labels == 0).sum()
        attacker = (labels == 1).sum()
        print(f"  正常用户: {normal} ({normal / len(labels) * 100:.1f}%)")
        print(f"  攻击者: {attacker} ({attacker / len(labels) * 100:.1f}%)")

        # 特征统计
        print("\n【特征统计】")
        print(f"  用户特征维度: {dataset['user_features'].shape[1]}")
        print(f"  问题特征维度: {dataset['question_features'].shape[1]}")
        print(f"  答案特征维度: {dataset['answer_features'].shape[1]}")

        # 攻击者特征对比
        print("\n【攻击者特征对比】")
        attacker_idx = dataset['user_labels'] == 1
        normal_idx = dataset['user_labels'] == 0

        print("  特征              正常用户均值    攻击者均值")
        print("  " + "-" * 50)
        for i in range(3):  # 只对比前3个特征
            normal_mean = dataset['user_features'][normal_idx, i].mean()
            attacker_mean = dataset['user_features'][attacker_idx, i].mean()
            print(f"  feat[{i}]            {normal_mean:.4f}        {attacker_mean:.4f}")


# ==================== 15. 主函数 ====================
def main():
    """
    主函数：生成并保存数据集
    """
    print("=" * 60)
    print("问答系统异构图数据集生成器")
    print("=" * 60)

    # 创建数据生成器
    generator = DataGenerator()

    # 生成数据集
    dataset = generator.generate_dataset()

    # 打印统计信息
    generator.print_statistics(dataset)

    # 保存数据集
    generator.save_dataset(dataset, 'hetero_dataset.pkl')

    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    dataset = main()
