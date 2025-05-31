# -*- coding: utf-8 -*-
"""
图差异计算器 - 计算两个图之间的各种差异度量
"""

import re
from typing import Dict, List
import networkx as nx
import numpy as np


class GraphDifferenceCalculator:
    """图差异计算器 - 计算两个图之间的各种差异度量"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
    
    def node_difference_metric(self, cognitive_graph: nx.Graph, real_graph: nx.Graph) -> Dict:
        """计算节点差异度量"""
        cognitive_nodes = set(cognitive_graph.nodes())
        real_nodes = set(real_graph.nodes())
        
        # 节点数量差异
        node_count_diff = abs(len(cognitive_nodes) - len(real_nodes))
        
        # 节点重叠度 (Jaccard相似度)
        intersection = cognitive_nodes.intersection(real_nodes)
        union = cognitive_nodes.union(real_nodes)
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # 缺失节点和幻觉节点
        missing_nodes = real_nodes - cognitive_nodes  # agent未认知到的真实节点
        hallucinated_nodes = cognitive_nodes - real_nodes  # agent认知中的虚假节点
        
        return {
            'node_count_diff': node_count_diff,
            'jaccard_similarity': jaccard_similarity,
            'missing_nodes_count': len(missing_nodes),
            'hallucinated_nodes_count': len(hallucinated_nodes),
            'missing_nodes': list(missing_nodes),
            'hallucinated_nodes': list(hallucinated_nodes),
            'cognitive_nodes_count': len(cognitive_nodes),
            'real_nodes_count': len(real_nodes)
        }
    
    def edge_difference_metric(self, cognitive_graph: nx.Graph, real_graph: nx.Graph) -> Dict:
        """计算边差异度量"""
        cognitive_edges = set(cognitive_graph.edges())
        real_edges = set(real_graph.edges())
        
        # 边数量差异
        edge_count_diff = abs(len(cognitive_edges) - len(real_edges))
        
        # 边重叠度
        intersection = cognitive_edges.intersection(real_edges)
        union = cognitive_edges.union(real_edges)
        edge_jaccard = len(intersection) / len(union) if union else 0
        
        # 缺失边和幻觉边
        missing_edges = real_edges - cognitive_edges
        hallucinated_edges = cognitive_edges - real_edges
        
        return {
            'edge_count_diff': edge_count_diff,
            'edge_jaccard_similarity': edge_jaccard,
            'missing_edges_count': len(missing_edges),
            'hallucinated_edges_count': len(hallucinated_edges),
            'missing_edges': list(missing_edges),
            'hallucinated_edges': list(hallucinated_edges),
            'cognitive_edges_count': len(cognitive_edges),
            'real_edges_count': len(real_edges)
        }
    
    def calculate_semantic_accuracy(self, cognitive_graph: nx.Graph, real_graph: nx.Graph) -> float:
        """计算语义准确性 (基于边权重差异)"""
        common_edges = set(cognitive_graph.edges()).intersection(set(real_graph.edges()))
        
        if not common_edges:
            return 0.0
        
        weight_differences = []
        for edge in common_edges:
            cognitive_weight = cognitive_graph[edge[0]][edge[1]].get('weight', 1)
            real_weight = real_graph[edge[0]][edge[1]].get('weight', 1)
            
            # 计算相对差异
            max_weight = max(cognitive_weight, real_weight)
            if max_weight > 0:
                relative_diff = abs(cognitive_weight - real_weight) / max_weight
                weight_differences.append(1 - relative_diff)  # 转换为相似度
        
        return np.mean(weight_differences) if weight_differences else 0.0
    
    def analyze_memory_reality_consistency_with_llm(self, agent_memories: List[str], 
                                                   real_conversations: List[str]) -> Dict:
        """使用LLM分析记忆与真实对话的一致性"""
        # 记忆一致性分析已禁用，因为没有提供具体记忆内容无法进行深层分析
        return {
            'consistency_score': 0.0, 
            'analysis': '记忆一致性分析已禁用，因为没有提供具体记忆内容无法进行深层分析'
        }
    
    def _extract_consistency_score(self, llm_response: str) -> float:
        """从LLM响应中提取一致性评分"""
        try:
            # 查找评分模式
            score_patterns = [
                r'一致性评分[：:](\\d*\\.?\\d+)',
                r'评分[：:](\\d*\\.?\\d+)',
                r'(\\d*\\.?\\d+)分',
                r'(0\.[0-9]+|1\.0|1|0)'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, llm_response)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # 确保在0-1范围内
            
            # 如果没有找到数字评分，尝试从文本中推断
            if '完全一致' in llm_response or '高度一致' in llm_response:
                return 0.9
            elif '基本一致' in llm_response or '大部分一致' in llm_response:
                return 0.7
            elif '部分一致' in llm_response:
                return 0.5
            elif '不太一致' in llm_response:
                return 0.3
            elif '完全不一致' in llm_response:
                return 0.1
            
            return 0.5  # 默认值
            
        except Exception as e:
            print(f"提取一致性评分时出错: {e}")
            return 0.0
    
    def _extract_analysis_reason(self, llm_response: str) -> str:
        """从LLM响应中提取分析理由"""
        try:
            # 查找分析理由
            reason_patterns = [
                r'分析理由[：:](.*?)(?=\n|$)',
                r'理由[：:](.*?)(?=\n|$)',
                r'说明[：:](.*?)(?=\n|$)'
            ]
            
            for pattern in reason_patterns:
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # 如果没有找到特定格式，返回整个响应的简化版本
            lines = llm_response.split('\n')
            for line in lines:
                if len(line.strip()) > 10 and '评分' not in line:
                    return line.strip()[:200]  # 返回第一个有意义的行
            
            return llm_response[:200]  # 返回前200个字符
            
        except Exception as e:
            print(f"提取分析理由时出错: {e}")
            return "无法提取分析理由"
    
    def compare_interaction_patterns_with_llm(self, cognitive_interactions: Dict, 
                                            real_interactions: Dict) -> Dict:
        """使用LLM比较交互模式的相似性"""
        if not self.llm_model:
            return {'pattern_similarity': 0.0, 'analysis': '无LLM模型'}
        
        try:
            print(f"开始LLM交互模式比较分析...")
            print(f"认知交互模式数量: {len(cognitive_interactions)}")
            print(f"真实交互模式数量: {len(real_interactions)}")
            
            # 构建交互模式描述
            cognitive_desc = self._describe_interaction_patterns(cognitive_interactions)
            real_desc = self._describe_interaction_patterns(real_interactions)
            
            prompt = f"""请比较以下两种交互模式的相似性：

**Agent认知中的交互模式：**
{cognitive_desc}

**真实的交互模式：**
{real_desc}

请分析：
1. 交互频率的相似性
2. 交互对象的相似性
3. 交互内容类型的相似性
4. 交互时机的相似性

请给出一个0-1之间的相似性评分（1表示完全相似，0表示完全不同），并简要说明理由。

输出格式：
相似性评分：[0.0-1.0]
分析理由：[简要说明]"""
            
            print(f"正在调用LLM模型进行交互模式比较...")
            response = self.llm_model.completion(prompt, temperature=0.1)
            
            print(f"LLM交互模式比较响应长度: {len(response) if response else 0}")
            if response:
                pass
            
            if not response:
                return {'pattern_similarity': 0.0, 'analysis': 'LLM响应为空'}
            
            similarity_score = self._extract_consistency_score(response)  # 复用评分提取方法
            analysis = self._extract_analysis_reason(response)
            
            return {
                'pattern_similarity': similarity_score,
                'analysis': analysis,
                'llm_response': response
            }
            
        except Exception as e:
            print(f"LLM交互模式比较时出错: {e}")
            return {'pattern_similarity': 0.0, 'analysis': f'比较出错: {str(e)}'}
    
    def _extract_interaction_patterns(self, graph: nx.Graph) -> Dict:
        """从图中提取交互模式"""
        interactions = {}
        
        for edge in graph.edges(data=True):
            node1, node2, data = edge
            agent_pair = f"{node1}-{node2}"
            
            interactions[agent_pair] = {
                'weight': data.get('weight', 1),
                'chat_count': data.get('chat_count', 0),
                'event_count': data.get('event_count', 0),
                'interaction_type': data.get('type', 'unknown')
            }
        
        return interactions
    
    def _describe_interaction_patterns(self, interactions: Dict) -> str:
        """描述交互模式"""
        if not interactions:
            return "无交互记录"
        
        descriptions = []
        for agent_pair, data in list(interactions.items())[:5]:  # 限制数量
            if isinstance(data, dict):
                weight = data.get('weight', 0)
                chat_count = data.get('chat_count', 0)
                event_count = data.get('event_count', 0)
                descriptions.append(f"{agent_pair}: 权重{weight}, 对话{chat_count}次, 事件{event_count}次")
            else:
                descriptions.append(f"{agent_pair}: {data}")
        
        return "\n".join(descriptions)
    
    def calculate_temporal_consistency(self, cognitive_graph: nx.Graph, real_graph: nx.Graph) -> float:
        """计算时间一致性 (简化版本，基于图的连通性)"""
        # 简化的时间一致性度量：基于图的连通性和密度
        cognitive_density = nx.density(cognitive_graph) if cognitive_graph.number_of_nodes() > 1 else 0
        real_density = nx.density(real_graph) if real_graph.number_of_nodes() > 1 else 0
        
        if real_density == 0:
            return 1.0 if cognitive_density == 0 else 0.0
        
        density_similarity = 1 - abs(cognitive_density - real_density) / real_density
        return max(0, density_similarity)