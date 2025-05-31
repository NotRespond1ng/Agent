# -*- coding: utf-8 -*-
"""
Agent认知模型与真实世界差距度量分析器

该模块提供了一套完整的度量方法来衡量agent脑中认识的世界和真实世界的差距。
主要功能包括：
1. 从agent记忆中提取认知图
2. 从真实对话数据中提取真实世界图
3. 计算图结构差异度量
4. 生成综合认知差距报告
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import traceback
import networkx as nx
import numpy as np
from pathlib import Path


class CognitiveGraphExtractor:
    """认知图提取器 - 从agent记忆中提取认知关系图"""
    
    def __init__(self, llm_model=None):
        self.agent_name_pattern = re.compile(r'[\u4e00-\u9fff]+|[A-Za-z]+')  # 匹配中文名或英文名
        self.llm_model = llm_model  # LLM模型实例，用于智能识别Agent名称
    
    def decode_text(self, text: str) -> str:
        """解码可能包含Unicode转义序列的文本"""
        if not text:
            return text
            
        try:
            # 处理Unicode转义序列
            if '\\u' in text:
                # 先尝试直接解码
                decoded = text.encode().decode('unicode_escape')
                return decoded
            return text
        except Exception as e:
            print(f"解码文本时出错: {e}, 原文本: {text[:100]}...")
            return text
    
    def validate_agent_name(self, name: str, context: str) -> bool:
        """验证提取的名称是否真的是Agent名称"""
        if len(name) < 2 or len(name) > 10:  # 名称长度合理性检查
            return False
            
        # 检查是否在对话上下文中出现
        dialogue_indicators = ['说', '问', '回答', '告诉', '聊天', '对话', '交谈', 
                              'said', 'asked', 'replied', 'told', 'chat', 'talk']
        
        # 在上下文中查找名称周围是否有对话指示词
        name_index = context.find(name)
        if name_index != -1:
            # 检查名称前后的文本
            start = max(0, name_index - 20)
            end = min(len(context), name_index + len(name) + 20)
            surrounding_text = context[start:end]
            
            for indicator in dialogue_indicators:
                if indicator in surrounding_text:
                    return True
        
        return False
    
    def extract_agents_from_text(self, text: str) -> Set[str]:
        """从文本中提取agent名称"""
        # 首先解码文本
        decoded_text = self.decode_text(text)
        
        # 如果有LLM模型，使用智能识别
        if self.llm_model:
            return self.extract_agents_with_llm(decoded_text)
        
        # 否则使用传统的正则表达式方法
        return self.extract_agents_with_regex(decoded_text)
    
    def extract_agents_with_llm(self, text: str) -> Set[str]:
        """使用LLM智能识别Agent名称"""
        try:
            # 构建提示词
            prompt = f"""请从以下文本中识别出所有的人物名称（Agent名称）。这些名称可能是中文名字或英文名字。
请只返回人物名称，每个名称占一行，不要包含其他内容。如果没有找到人物名称，请返回"无"。

文本内容：
{text[:500]}  # 限制文本长度避免token过多

人物名称："""
            
            # 调用LLM
            response = self.llm_model.completion(prompt, temperature=0.1)
            
            if not response or response.strip() == "无":
                return set()
            
            # 解析LLM返回的结果
            agent_names = set()
            lines = response.strip().split('\n')
            
            for line in lines:
                name = line.strip()
                # 基本过滤：长度检查和常见词汇过滤
                if (len(name) >= 2 and len(name) <= 10 and 
                    not any(word in name for word in ['无', '没有', '不存在', 'None', 'No']) and
                    re.match(r'^[\u4e00-\u9fff\w]+$', name)):  # 只包含中文、字母、数字
                    agent_names.add(name)
            
            print(f"LLM识别到的Agent名称: {agent_names}")
            return agent_names
            
        except Exception as e:
            print(f"LLM识别Agent名称时出错: {e}，回退到正则表达式方法")
            return self.extract_agents_with_regex(text)
    
    def extract_agents_with_regex(self, text: str) -> Set[str]:
        """使用正则表达式提取Agent名称（传统方法）"""
        # 常见的agent名称模式
        potential_names = self.agent_name_pattern.findall(text)
        
        # 过滤掉常见的非名称词汇
        common_words = {'正在', '此时', '对话', '聊天', '说话', '交谈', '讨论', '谈论', '回答', '告诉',
                       '现在', '今天', '昨天', '明天', '时候', '地方', '房间', '家里', '外面', '里面',
                       'is', 'was', 'are', 'were', 'the', 'and', 'or', 'but', 'chat', 'talk',
                       'said', 'asked', 'replied', 'told', 'now', 'today', 'yesterday', 'tomorrow'}
        
        agent_names = set()
        for name in potential_names:
            if (len(name) >= 2 and 
                name not in common_words and 
                self.validate_agent_name(name, text)):
                agent_names.add(name)
        
        return agent_names
    
    def extract_cognitive_graph(self, agent) -> nx.Graph:
        """从agent的记忆中提取认知图 - 使用向量检索而不是memory.json"""
        G = nx.Graph()
        G.add_node(agent.name)  # 添加自己作为中心节点
        
        print(f"\n=== {agent.name} 记忆分析 ===")
        
        # 获取LLM模型实例（如果agent有的话）
        llm_model = None
        try:
            # 尝试从agent的llm属性中获取LLM模型
            if hasattr(agent, 'llm') and agent.llm:
                llm_model = agent.llm
                print(f"成功获取到Agent的LLM模型: {type(llm_model)}")
            # 或者尝试从其他可能的属性中获取
            elif hasattr(agent, 'llm_model') and agent.llm_model:
                llm_model = agent.llm_model
                print(f"从llm_model属性获取到LLM模型: {type(llm_model)}")
            else:
                print("未找到LLM模型实例，将使用传统的正则表达式方法")
                print(f"Agent属性: {[attr for attr in dir(agent) if not attr.startswith('_')]}")
        except Exception as e:
            print(f"获取LLM模型时出错: {e}，将使用传统方法")
            print(f"错误详情: {traceback.format_exc()}")
        
        # 创建认知图提取器，传入LLM模型
        extractor = CognitiveGraphExtractor(llm_model=llm_model)
        
        # 使用向量检索获取对话相关的记忆
        chat_queries = [
            "对话", "聊天", "说话", "交谈", "讨论", "谈论", "回答", "告诉",
            "chat", "talk", "conversation", "dialogue", "speak", "say"
        ]
        
        chat_agents_found = set()
        chat_memories_count = 0
        
        for query in chat_queries:
            try:
                # 使用Associate的向量检索功能
                chat_nodes = agent.associate._retrieve_nodes('chat', query, top_k=10)
                
                for node in chat_nodes:
                    if hasattr(node, 'text') and node.text:
                        chat_memories_count += 1
                        
                        # 显示前几个记忆的内容（调试用）
                        if chat_memories_count <= 3:
                            original_text = node.text[:100]
                            decoded_text = self.decode_text(node.text)[:100]
                            print(f"\n对话记忆 {chat_memories_count} (查询: {query}):")
                            print(f"原始文本: {original_text}...")
                            print(f"解码文本: {decoded_text}...")
                        
                        other_agents = extractor.extract_agents_from_text(node.text)
                        chat_agents_found.update(other_agents)
                        
                        # 显示提取到的Agent（调试用）
                        if chat_memories_count <= 3 and other_agents:
                            print(f"提取到的Agent: {other_agents}")
                        
                        for other_agent in other_agents:
                            if other_agent != agent.name:
                                G.add_node(other_agent)
                                if G.has_edge(agent.name, other_agent):
                                    G[agent.name][other_agent]['weight'] += 1
                                    G[agent.name][other_agent]['chat_count'] += 1
                                else:
                                    G.add_edge(agent.name, other_agent, 
                                             weight=1, chat_count=1, event_count=0, type='chat')
            except Exception as e:
                print(f"检索对话记忆时出错 (查询: {query}): {e}")
                continue
        
        # 使用向量检索获取事件相关的记忆
        event_queries = [
            "事件", "发生", "经历", "体验", "活动", "行为", "动作",
            "event", "happen", "occur", "experience", "activity", "action"
        ]
        
        event_agents_found = set()
        event_memories_count = 0
        
        for query in event_queries:
            try:
                # 使用Associate的向量检索功能
                event_nodes = agent.associate._retrieve_nodes('event', query, top_k=10)
                
                for node in event_nodes:
                    if hasattr(node, 'text') and node.text:
                        event_memories_count += 1
                        
                        # 显示前几个记忆的内容（调试用）
                        if event_memories_count <= 3:
                            original_text = node.text[:100]
                            decoded_text = self.decode_text(node.text)[:100]
                            print(f"\n事件记忆 {event_memories_count} (查询: {query}):")
                            print(f"原始文本: {original_text}...")
                            print(f"解码文本: {decoded_text}...")
                        
                        related_agents = extractor.extract_agents_from_text(node.text)
                        event_agents_found.update(related_agents)
                        
                        # 显示提取到的Agent（调试用）
                        if event_memories_count <= 3 and related_agents:
                            print(f"提取到的Agent: {related_agents}")
                        
                        for related_agent in related_agents:
                            if related_agent != agent.name:
                                G.add_node(related_agent)
                                if G.has_edge(agent.name, related_agent):
                                    G[agent.name][related_agent]['weight'] += 0.5  # 事件权重较低
                                    G[agent.name][related_agent]['event_count'] += 1
                                else:
                                    G.add_edge(agent.name, related_agent, 
                                             weight=0.5, chat_count=0, event_count=1, type='event')
            except Exception as e:
                print(f"检索事件记忆时出错 (查询: {query}): {e}")
                continue
        
        # 调试信息：显示提取结果
        all_agents_found = chat_agents_found.union(event_agents_found)
        print(f"\n=== 提取结果统计 ===")
        print(f"对话记忆检索次数: {chat_memories_count}")
        print(f"事件记忆检索次数: {event_memories_count}")
        print(f"从对话记忆中提取到的Agent: {chat_agents_found}")
        print(f"从事件记忆中提取到的Agent: {event_agents_found}")
        print(f"总共提取到的Agent: {all_agents_found}")
        print(f"认知图节点数: {G.number_of_nodes()}")
        print(f"认知图边数: {G.number_of_edges()}")
        
        return G


class RealWorldGraphExtractor:
    """真实世界图提取器 - 从对话数据中提取真实关系图"""
    
    def extract_participants(self, location_key: str) -> List[str]:
        """从位置键中提取参与者"""
        if " @ " in location_key:
            participants_part = location_key.split(" @ ")[0]
        else:
            participants_part = location_key
        
        participants_raw = participants_part.split(" -> ")
        participants = [name.strip() for name in participants_raw if name.strip()]
        
        return participants
    
    def extract_real_world_graph(self, conversation_data: Dict) -> nx.Graph:
        """从真实对话数据中提取真实世界图"""
        G = nx.Graph()
        
        if not conversation_data:
            return G
        
        for timestamp, chats_at_time in conversation_data.items():
            if not isinstance(chats_at_time, list):
                continue
            
            for chat_group in chats_at_time:
                if not isinstance(chat_group, dict) or len(chat_group.keys()) != 1:
                    continue
                
                location_key = list(chat_group.keys())[0]
                chat_log = chat_group[location_key]
                
                try:
                    participants = self.extract_participants(location_key)
                    
                    if len(participants) < 2:
                        continue
                    
                    # 添加节点和边
                    for i, agent1 in enumerate(participants):
                        G.add_node(agent1)
                        for agent2 in participants[i+1:]:
                            G.add_node(agent2)
                            
                            # 计算对话轮数作为权重
                            chat_rounds = len(chat_log) if isinstance(chat_log, list) else 1
                            
                            if G.has_edge(agent1, agent2):
                                G[agent1][agent2]['weight'] += chat_rounds
                                G[agent1][agent2]['interaction_count'] += 1
                            else:
                                G.add_edge(agent1, agent2, 
                                         weight=chat_rounds, interaction_count=1)
                
                except Exception as e:
                    print(f"处理对话组时出错: {e}")
                    continue
        
        return G


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
        if not self.llm_model:
            print("未提供LLM模型，无法进行语义一致性分析")
            return {'consistency_score': 0.0, 'analysis': '无LLM模型'}
        
        try:
            # 构建分析提示词
            memories_text = "\n".join([f"记忆{i+1}: {mem[:200]}" for i, mem in enumerate(agent_memories[:5])])
            conversations_text = "\n".join([f"对话{i+1}: {conv[:200]}" for i, conv in enumerate(real_conversations[:5])])
            
            prompt = f"""请分析以下Agent的记忆内容与真实对话记录的一致性程度。

**Agent记忆内容：**
{memories_text}

**真实对话记录：**
{conversations_text}

请从以下几个维度进行分析：
1. 事实一致性：记忆中的事实是否与真实对话一致
2. 情感一致性：记忆中的情感表达是否与真实对话一致
3. 时间一致性：记忆中的时间顺序是否与真实对话一致
4. 人物关系一致性：记忆中的人物关系是否与真实对话一致

请给出一个0-1之间的一致性评分（1表示完全一致，0表示完全不一致），并简要说明理由。

输出格式：
一致性评分：[0.0-1.0]
分析理由：[简要说明]"""
            
            # 调用LLM
            response = self.llm_model.completion(prompt, temperature=0.1)
            
            if not response:
                return {'consistency_score': 0.0, 'analysis': 'LLM响应为空'}
            
            # 解析LLM响应
            consistency_score = self._extract_consistency_score(response)
            analysis = self._extract_analysis_reason(response)
            
            return {
                'consistency_score': consistency_score,
                'analysis': analysis,
                'llm_response': response
            }
            
        except Exception as e:
            print(f"LLM语义一致性分析时出错: {e}")
            return {'consistency_score': 0.0, 'analysis': f'分析出错: {str(e)}'}
    
    def _extract_consistency_score(self, llm_response: str) -> float:
        """从LLM响应中提取一致性评分"""
        try:
            import re
            # 查找评分模式
            score_patterns = [
                r'一致性评分[：:](\d*\.?\d+)',
                r'评分[：:](\d*\.?\d+)',
                r'(\d*\.?\d+)分',
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
            import re
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
            
            response = self.llm_model.completion(prompt, temperature=0.1)
            
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


class CognitiveWorldGapAnalyzer:
    """认知世界差距分析器 - 主要分析类"""
    
    def __init__(self, agents: Dict = None, conversation_data: Dict = None, llm_model=None):
        self.agents = agents or {}
        self.conversation_data = conversation_data or {}
        self.llm_model = llm_model
        
        self.cognitive_extractor = CognitiveGraphExtractor(llm_model=llm_model)
        self.real_world_extractor = RealWorldGraphExtractor()
        self.diff_calculator = GraphDifferenceCalculator(llm_model=llm_model)
        
        # 提取真实世界图
        self.real_world_graph = self.real_world_extractor.extract_real_world_graph(
            self.conversation_data
        )
    
    def load_conversation_data(self, filepath: str) -> bool:
        """加载对话数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_data = json.load(f)
            
            # 重新提取真实世界图
            self.real_world_graph = self.real_world_extractor.extract_real_world_graph(
                self.conversation_data
            )
            return True
        except Exception as e:
            print(f"加载对话数据失败: {e}")
            return False
    
    def get_agent_real_subgraph(self, agent_name: str) -> nx.Graph:
        """获取与特定agent相关的真实世界子图"""
        if agent_name not in self.real_world_graph:
            return nx.Graph()
        
        # 获取该agent的邻居节点
        neighbors = list(self.real_world_graph.neighbors(agent_name))
        subgraph_nodes = [agent_name] + neighbors
        
        return self.real_world_graph.subgraph(subgraph_nodes).copy()
    
    def cognitive_world_gap_metric(self, cognitive_graph: nx.Graph, real_graph: nx.Graph, 
                                 weights: Dict = None) -> Dict:
        """综合认知差距度量"""
        if weights is None:
            weights = {
                'node_structure': 0.3,
                'edge_structure': 0.4,
                'semantic_accuracy': 0.2,
                'temporal_consistency': 0.1
            }
        
        # 1. 结构差异
        node_metrics = self.diff_calculator.node_difference_metric(cognitive_graph, real_graph)
        edge_metrics = self.diff_calculator.edge_difference_metric(cognitive_graph, real_graph)
        
        # 节点结构得分 (0-1, 1表示完全一致)
        node_structure_score = node_metrics['jaccard_similarity']
        
        # 边结构得分
        edge_structure_score = edge_metrics['edge_jaccard_similarity']
        
        # 2. 语义准确性
        semantic_accuracy = self.diff_calculator.calculate_semantic_accuracy(
            cognitive_graph, real_graph
        )
        
        # 3. 时间一致性
        temporal_consistency = self.diff_calculator.calculate_temporal_consistency(
            cognitive_graph, real_graph
        )
        
        # 综合得分计算
        overall_score = (
            weights['node_structure'] * node_structure_score +
            weights['edge_structure'] * edge_structure_score +
            weights['semantic_accuracy'] * semantic_accuracy +
            weights['temporal_consistency'] * temporal_consistency
        )
        
        # 差距度量 (0-1, 0表示完全一致，1表示完全不同)
        cognitive_gap = 1 - overall_score
        
        return {
            'cognitive_gap': cognitive_gap,
            'overall_score': overall_score,
            'node_structure_score': node_structure_score,
            'edge_structure_score': edge_structure_score,
            'semantic_accuracy': semantic_accuracy,
            'temporal_consistency': temporal_consistency,
            'detailed_metrics': {
                'nodes': node_metrics,
                'edges': edge_metrics
            }
        }
    
    def analyze_agent_cognitive_gap(self, agent_name: str) -> Dict:
        """分析单个agent的认知差距"""
        if agent_name not in self.agents:
            return {'error': f'Agent {agent_name} not found'}
        
        agent = self.agents[agent_name]
        
        # 提取认知图
        cognitive_graph = self.cognitive_extractor.extract_cognitive_graph(agent)
        
        # 提取该agent相关的真实世界子图
        real_subgraph = self.get_agent_real_subgraph(agent_name)
        
        # 计算差距度量
        gap_metrics = self.cognitive_world_gap_metric(cognitive_graph, real_subgraph)
        
        # 使用LLM进行深度分析（如果可用）
        llm_analysis = {}
        if self.llm_model:
            try:
                # 分析记忆与真实对话的一致性
                memory_consistency = self.diff_calculator.analyze_memory_reality_consistency_with_llm(
                    agent, self.conversation_data
                )
                
                # 分析交互模式的相似性
                interaction_similarity = self.diff_calculator.compare_interaction_patterns_with_llm(
                    cognitive_graph, real_subgraph
                )
                
                llm_analysis = {
                    'memory_consistency': memory_consistency,
                    'interaction_similarity': interaction_similarity
                }
                
                print(f"LLM分析完成 - Agent: {agent_name}")
                
            except Exception as e:
                print(f"LLM分析失败 - Agent: {agent_name}, 错误: {str(e)}")
                llm_analysis = {'error': str(e)}
        
        return {
            'agent_name': agent_name,
            'cognitive_nodes': len(cognitive_graph.nodes()),
            'cognitive_edges': len(cognitive_graph.edges()),
            'real_nodes': len(real_subgraph.nodes()),
            'real_edges': len(real_subgraph.edges()),
            'gap_metrics': gap_metrics,
            'llm_analysis': llm_analysis,
            'cognitive_graph_info': {
                'nodes': list(cognitive_graph.nodes()),
                'edges': [(u, v, d) for u, v, d in cognitive_graph.edges(data=True)]
            },
            'real_graph_info': {
                'nodes': list(real_subgraph.nodes()),
                'edges': [(u, v, d) for u, v, d in real_subgraph.edges(data=True)]
            }
        }
    
    def analyze_all_agents(self) -> Dict:
        """分析所有agent的认知差距"""
        results = {}
        for agent_name in self.agents.keys():
            results[agent_name] = self.analyze_agent_cognitive_gap(agent_name)
        
        return results
    
    def generate_gap_report(self) -> Dict:
        """生成认知差距报告"""
        all_results = self.analyze_all_agents()
        
        # 过滤掉错误结果
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                'error': 'No valid agent analysis results',
                'total_agents': len(self.agents),
                'analyzed_agents': 0
            }
        
        # 计算统计信息
        cognitive_gaps = [r['gap_metrics']['cognitive_gap'] for r in valid_results.values()]
        
        max_gap_item = max(valid_results.items(), key=lambda x: x[1]['gap_metrics']['cognitive_gap'])
        min_gap_item = min(valid_results.items(), key=lambda x: x[1]['gap_metrics']['cognitive_gap'])
        
        report = {
            'summary': {
                'total_agents': len(self.agents),
                'analyzed_agents': len(valid_results),
                'average_cognitive_gap': np.mean(cognitive_gaps),
                'std_cognitive_gap': np.std(cognitive_gaps),
                'max_gap_agent': {
                    'name': max_gap_item[0],
                    'gap': max_gap_item[1]['gap_metrics']['cognitive_gap']
                },
                'min_gap_agent': {
                    'name': min_gap_item[0],
                    'gap': min_gap_item[1]['gap_metrics']['cognitive_gap']
                },
                'real_world_graph_info': {
                    'total_nodes': len(self.real_world_graph.nodes()),
                    'total_edges': len(self.real_world_graph.edges()),
                    'density': nx.density(self.real_world_graph)
                }
            },
            'detailed_results': valid_results
        }
        
        return report
    
    def save_report(self, report: Dict, filepath: str) -> bool:
        """保存报告到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存报告失败: {e}")
            return False
    
    def print_summary(self, report: Dict = None):
        """打印报告摘要"""
        if report is None:
            report = self.generate_gap_report()
        
        if 'error' in report:
            print(f"错误: {report['error']}")
            return
        
        summary = report['summary']
        print("\n=== Agent认知差距分析报告 ===")
        print(f"总Agent数量: {summary['total_agents']}")
        print(f"成功分析的Agent数量: {summary['analyzed_agents']}")
        print(f"平均认知差距: {summary['average_cognitive_gap']:.3f}")
        print(f"认知差距标准差: {summary['std_cognitive_gap']:.3f}")
        print(f"认知差距最大的Agent: {summary['max_gap_agent']['name']} (差距: {summary['max_gap_agent']['gap']:.3f})")
        print(f"认知差距最小的Agent: {summary['min_gap_agent']['name']} (差距: {summary['min_gap_agent']['gap']:.3f})")
        print(f"\n真实世界图信息:")
        print(f"  节点数: {summary['real_world_graph_info']['total_nodes']}")
        print(f"  边数: {summary['real_world_graph_info']['total_edges']}")
        print(f"  密度: {summary['real_world_graph_info']['density']:.3f}")
        print("\n=== 详细分析 ===")
        
        for agent_name, result in report['detailed_results'].items():
            gap = result['gap_metrics']['cognitive_gap']
            print(f"{agent_name}: 认知差距={gap:.3f}, "
                  f"认知图({result['cognitive_nodes']}节点,{result['cognitive_edges']}边), "
                  f"真实图({result['real_nodes']}节点,{result['real_edges']}边)")


def load_agent_from_storage(agent_name: str, storage_path: str, config: Dict = None) -> 'Agent':
    """从存储路径加载agent"""
    try:
        # 导入必要的模块
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from modules.agent import Agent
        from modules import memory
        
        # 从agent.json文件读取配置
        if config is None:
            agent_config_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
            
            if not os.path.exists(agent_config_path):
                print(f"找不到agent配置文件: {agent_config_path}")
                return None
            
            try:
                with open(agent_config_path, 'r', encoding='utf-8') as f:
                    agent_config = json.load(f)
            except Exception as e:
                print(f"读取agent配置文件失败: {e}")
                return None
            
            # 确保agent_config是字典类型
            if not isinstance(agent_config, dict):
                print(f"agent配置文件格式错误，期望字典类型，实际为: {type(agent_config)}")
                return None
            
            # 构建完整的配置
            config = {
                "name": agent_config.get("name", agent_name),
                "storage_root": storage_path,
                "percept": {"vision_r": 3, "att_bandwidth": 3},
                "think": {"interval": 10},
                "chat_iter": 3,
                "spatial": agent_config.get("spatial", {
                    "tree": {"世界": {"房间": ["客厅", "卧室"]}},
                    "address": {"living_area": ["世界", "房间", "客厅"]}
                }),
                "schedule": {
                    "daily_schedule": [],
                    "diversity": 5,
                    "max_try": 5
                },
                "associate": {
                    "embedding": {
                        "type": "openai",
                        "model": "text-embedding-3-small",
                        "base_url": "https://yunwu.ai/v1"
                    },
                    "retention": 8,
                    "max_memory": -1,
                    "max_importance": 10
                },
                "llm": {
                    "type": "openai",
                    "model": "gpt-4o-mini",
                    "base_url": "https://yunwu.ai/v1"
                },
                "currently": agent_config.get("currently", f"{agent_name}正在思考"),
                "scratch": agent_config.get("scratch", {
                    "age": 25,
                    "innate": "友好、好奇",
                    "learned": "这是一个虚拟的agent",
                    "lifestyle": "正常的生活节奏",
                    "daily_plan": ""
                }),
                "coord": agent_config.get("coord", [0, 0])
            }
        
        # 创建虚拟的maze和conversation对象
        class DummyTile:
            def __init__(self, coord):
                self.coord = coord
                self.address = ["世界", "房间", "客厅", "物品"]
                self.address_keys = ["world", "sector", "arena", "game_object"]
                self.address_map = dict(zip(self.address_keys, self.address))
                self._events = {}
                self.event_cnt = 0
                self.collision = False
                
            def get_address(self, level=None, as_list=True):
                level = level or self.address_keys[-1]
                if level not in self.address_keys:
                    return self.address if as_list else ":".join(self.address)
                pos = self.address_keys.index(level) + 1
                if as_list:
                    return self.address[:pos]
                return ":".join(self.address[:pos])
                
            def get_addresses(self):
                addresses = []
                if len(self.address) > 1:
                    addresses = [
                        ":".join(self.address[:i]) for i in range(2, len(self.address) + 1)
                    ]
                return addresses
                
            def abstract(self):
                address = ":".join(self.address)
                if self.collision:
                    address += "(collision)"
                return {
                    "coord[{},{}]".format(self.coord[0], self.coord[1]): address,
                    "events": {k: str(v) for k, v in self.events.items()},
                }
                
            def get_events(self):
                return self.events.values()
                
            def update_events(self, event, match="subject"):
                u_events = {}
                for tag, eve in self._events.items():
                    if match == "subject" and hasattr(eve, 'subject') and hasattr(event, 'subject') and eve.subject == event.subject:
                        self._events[tag] = event
                        u_events[tag] = event
                return u_events
                
            def add_event(self, event):
                if all(e != event for e in self._events.values()):
                    self._events["e_" + str(self.event_cnt)] = event
                    self.event_cnt += 1
                return event
                
            def remove_events(self, subject=None, event=None):
                r_events = {}
                for tag, eve in self._events.items():
                    if subject and hasattr(eve, 'subject') and eve.subject == subject:
                        r_events[tag] = eve
                    if event and eve == event:
                        r_events[tag] = eve
                for r_eve in r_events:
                    self._events.pop(r_eve)
                return r_events
                
            def has_address(self, key):
                return key in self.address_map
                
            @property
            def events(self):
                return self._events
                
            @property
            def is_empty(self):
                return len(self.address) == 1 and not self._events
        
        class DummyMaze:
            def __init__(self):
                self.tiles = {}
                self.maze_width = 10
                self.maze_height = 10
                self.address_tiles = {}
                
            def tile_at(self, coord):
                coord_key = (coord[0], coord[1])
                if coord_key not in self.tiles:
                    self.tiles[coord_key] = DummyTile(coord)
                return self.tiles[coord_key]
                
            def get_address_tiles(self, address):
                addr = ":".join(address) if isinstance(address, list) else address
                if addr in self.address_tiles:
                    return self.address_tiles[addr]
                return {(0, 0)}
                
            def find_path(self, src_coord, dst_coord):
                # 简单的路径查找实现
                return [dst_coord]
                
            def get_scope(self, coord, config):
                # 返回当前位置的tile
                return [self.tile_at(coord)]
                
            def get_around(self, coord, no_collision=True):
                coords = [
                    (coord[0] - 1, coord[1]),
                    (coord[0] + 1, coord[1]),
                    (coord[0], coord[1] - 1),
                    (coord[0], coord[1] + 1),
                ]
                if no_collision:
                    coords = [c for c in coords if not self.tile_at(c).collision]
                return coords
                
            def update_obj(self, coord, obj_event):
                # 简单的虚拟实现
                pass
        
        class DummyConversation:
            pass
        
        class DummyLogger:
            def info(self, msg):
                pass
        
        # 创建agent实例
        agent = Agent(config, DummyMaze(), DummyConversation(), DummyLogger())
        return agent
    except Exception as e:
        print(f"加载agent {agent_name} 失败: {e}")
        print(traceback.format_exc()) # 打印完整的堆栈跟踪
        return None


def select_simulation() -> Optional[str]:
    """选择模拟"""
    checkpoints_dir = "results/checkpoints"
    
    if not os.path.exists(checkpoints_dir):
        print(f"错误: 找不到checkpoints目录: {checkpoints_dir}")
        return None
    
    # 获取所有模拟目录
    simulations = []
    for item in os.listdir(checkpoints_dir):
        sim_path = os.path.join(checkpoints_dir, item)
        if os.path.isdir(sim_path) and os.path.exists(os.path.join(sim_path, "conversation.json")):
            simulations.append(item)
    
    if not simulations:
        print("错误: 没有找到任何有效的模拟")
        return None
    
    # 显示模拟列表
    print("\n=== 可用的模拟 ===")
    for i, sim in enumerate(simulations, 1):
        print(f"{i}. {sim}")
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择模拟 (1-{len(simulations)}) 或输入 'q' 退出: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(simulations):
                return simulations[choice_idx]
            else:
                print(f"无效选择，请输入 1-{len(simulations)} 之间的数字")
        except ValueError:
            print("无效输入，请输入数字")


def select_agents(simulation_name: str) -> Optional[List[str]]:
    """选择要分析的agents"""
    storage_dir = f"results/checkpoints/{simulation_name}/storage"
    
    if not os.path.exists(storage_dir):
        print(f"错误: 找不到storage目录: {storage_dir}")
        return None
    
    # 获取所有agent目录
    agents = []
    for item in os.listdir(storage_dir):
        agent_path = os.path.join(storage_dir, item)
        if os.path.isdir(agent_path):
            agents.append(item)
    
    if not agents:
        print("错误: 没有找到任何agent")
        return None
    
    # 显示agent列表
    print(f"\n=== {simulation_name} 中的Agents ===")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent}")
    print(f"{len(agents) + 1}. 全部分析")
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择要分析的agent (1-{len(agents) + 1}) 或输入 'q' 退出: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if choice_idx == len(agents):  # 选择全部
                return agents
            elif 0 <= choice_idx < len(agents):
                return [agents[choice_idx]]
            else:
                print(f"无效选择，请输入 1-{len(agents) + 1} 之间的数字")
        except ValueError:
            print("无效输入，请输入数字")


def main():
    """主函数 - 交互式认知差距分析"""
    print("\n=== Agent认知模型与真实世界差距度量分析器 ===")
    
    # 1. 选择模拟
    simulation_name = select_simulation()
    if not simulation_name:
        print("退出程序")
        return
    
    print(f"\n选择的模拟: {simulation_name}")
    
    # 2. 选择agents
    selected_agents = select_agents(simulation_name)
    if not selected_agents:
        print("退出程序")
        return
    
    print(f"选择的agents: {', '.join(selected_agents)}")
    
    # 3. 加载agents
    print("\n正在加载agents...")
    loaded_agents = {}
    llm_model = None
    
    for agent_name in selected_agents:
        storage_path = f"results/checkpoints/{simulation_name}/storage/{agent_name}"
        agent = load_agent_from_storage(agent_name, storage_path)
        
        if agent:
            loaded_agents[agent_name] = agent
            print(f"✓ 成功加载 {agent_name}")
            
            # 尝试获取LLM模型实例（从第一个成功加载的agent中获取）
            if llm_model is None:
                try:
                    if hasattr(agent, 'llm') and agent.llm:
                        llm_model = agent.llm
                        print(f"✓ 从 {agent_name} 获取到LLM模型实例")
                    elif hasattr(agent, 'llm_model') and agent.llm_model:
                        llm_model = agent.llm_model
                        print(f"✓ 从 {agent_name} 获取到LLM模型实例")
                except Exception as e:
                    print(f"从 {agent_name} 获取LLM模型失败: {str(e)}")
        else:
            print(f"✗ 加载 {agent_name} 失败")
    
    if not loaded_agents:
        print("没有成功加载任何agent，退出程序")
        return
    
    # 4. 加载对话数据并创建分析器
    conversation_path = f"results/checkpoints/{simulation_name}/conversation.json"
    analyzer = CognitiveWorldGapAnalyzer(agents=loaded_agents, llm_model=llm_model)
    
    if not analyzer.load_conversation_data(conversation_path):
        print(f"加载对话数据失败: {conversation_path}")
        return
    
    print(f"成功加载对话数据: {conversation_path}")
    
    if llm_model:
        print("✓ LLM增强分析已启用")
    else:
        print("⚠ 未找到LLM模型，将使用传统分析方法")
    
    # 5. 生成分析报告
    print("\n正在生成认知差距分析报告...")
    report = analyzer.generate_gap_report()
    
    # 6. 显示结果
    analyzer.print_summary(report)
    
    # 7. 保存报告
    report_path = f"results/checkpoints/{simulation_name}/cognitive_gap_report.json"
    if analyzer.save_report(report, report_path):
        print(f"\n报告已保存到: {report_path}")
    else:
        print("\n保存报告失败")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()