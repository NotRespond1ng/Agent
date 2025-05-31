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

    def __init__(self, known_agents=None, llm_model=None):
        self.agent_name_pattern = re.compile(r'[\u4e00-\u9fff]+|[A-Za-z]+')  # 匹配中文名或英文名
        self.llm_model = llm_model  # LLM模型实例，用于智能识别Agent名称
        self.known_agents = set(known_agents) if known_agents else set()  # 已知的agent名称列表

    def decode_text(self, text: str) -> str:
        """解码可能包含Unicode转义序列的文本"""
        if not text:
            return text

        try:
            if '\\u' in text:
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

        dialogue_indicators = ['说', '问', '回答', '告诉', '聊天', '对话', '交谈',
                               'said', 'asked', 'replied', 'told', 'chat', 'talk']

        name_index = context.find(name)
        if name_index != -1:
            start = max(0, name_index - 20)
            end = min(len(context), name_index + len(name) + 20)
            surrounding_text = context[start:end]

            for indicator in dialogue_indicators:
                if indicator in surrounding_text:
                    return True
        return False

    def extract_agents_from_text(self, text: str) -> Set[str]:
        """从文本中提取agent名称"""
        decoded_text = self.decode_text(text)

        if self.known_agents:
            return self.extract_agents_from_known_list(decoded_text)
        if self.llm_model:
            return self.extract_agents_with_llm(decoded_text)
        return self.extract_agents_with_regex(decoded_text)

    def extract_agents_with_llm(self, text: str) -> Set[str]:
        """使用LLM智能识别Agent名称"""
        try:
            prompt = f"""请从以下文本中识别出所有的人物名称（Agent名称）。这些名称可能是中文名字或英文名字。
请只返回人物名称，每个名称占一行，不要包含其他内容。如果没有找到人物名称，请返回"无"。

文本内容：
{text[:500]}

人物名称："""
            response = self.llm_model.completion(prompt, temperature=0.1)

            if not response or response.strip() == "无":
                return set()

            agent_names = set()
            lines = response.strip().split('\n')

            for line in lines:
                name = line.strip()
                if (len(name) >= 2 and len(name) <= 10 and
                        not any(word in name for word in ['无', '没有', '不存在', 'None', 'No']) and
                        re.match(r'^[\u4e00-\u9fff\w]+$', name)):
                    agent_names.add(name)
            print(f"LLM识别到的Agent名称: {agent_names}") # 核心：显示LLM识别结果
            return agent_names
        except Exception as e:
            print(f"LLM识别Agent名称时出错: {e}，回退到正则表达式方法")
            return self.extract_agents_with_regex(text)

    def extract_agents_from_known_list(self, text: str) -> Set[str]:
        """从已知agent列表中查找在文本中出现的agent名称（推荐方法）"""
        found_agents = set()
        for agent_name_to_find in self.known_agents:
            if agent_name_to_find in text:
                found_agents.add(agent_name_to_find)

        for agent_name in self.known_agents:
            if agent_name in text:
                pattern = r'\b' + re.escape(agent_name) + r'\b'
                if re.search(pattern, text):
                    found_agents.add(agent_name)
                elif agent_name in text:
                    found_agents.add(agent_name)
        # print(f"    [extract_agents_from_known_list] 从当前文本提取到的 agents: {found_agents}") # 保留此项用于确认单步提取结果
        return found_agents

    def extract_agents_with_regex(self, text: str) -> Set[str]:
        """使用正则表达式提取Agent名称（传统方法）"""
        potential_names = self.agent_name_pattern.findall(text)
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
        """从agent的记忆中提取认知图 - 使用向量检索"""
        G = nx.Graph()
        G.add_node(agent.name)  # 添加自己作为中心节点

        try:
            self._initialize_agent_memory(agent)
        except Exception as e:
            print(f"初始化Agent {agent.name} 记忆时出错: {e}")
            # print(f"错误详情: {traceback.format_exc()}") # 调试时可取消注释

        chat_queries = ["对话", "聊天", "说话", "交谈", "讨论", "谈论", "回答", "告诉",
                        "chat", "talk", "conversation", "dialogue", "speak", "say"]
        chat_agents_found = set()
        chat_memories_count = 0

        for query in chat_queries:
            try:
                chat_nodes = agent.associate._retrieve_nodes('chat', query)
                for node in chat_nodes:
                    if hasattr(node, 'describe') and node.describe:
                        chat_memories_count += 1
                        # print(f"  [extract_cognitive_graph] 正在分析对话记忆 (ID: {getattr(node, 'node_id', '未知ID')}): '{node.describe[:50]}...'") # 核心：显示正在分析的记忆
                        other_agents = self.extract_agents_from_text(node.describe)
                        # print(f"  [extract_cognitive_graph] 从对话记忆提取到的 agents: {other_agents}") # 核心：显示从该记忆中提取的agent
                        chat_agents_found.update(other_agents)
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
                print(f"检索 {agent.name} 的对话记忆时出错 (查询: {query}): {e}")
                continue

        event_queries = ["事件", "发生", "经历", "体验", "活动", "行为", "动作",
                         "event", "happen", "occur", "experience", "activity", "action"]
        event_agents_found = set()
        event_memories_count = 0

        for query in event_queries:
            try:
                event_nodes = agent.associate._retrieve_nodes('event', query)
                for node in event_nodes:
                    if hasattr(node, 'describe') and node.describe:
                        event_memories_count += 1
                        # print(f"  [extract_cognitive_graph] 正在分析事件记忆 (ID: {getattr(node, 'node_id', '未知ID')}): '{node.describe[:50]}...'") # 核心：显示正在分析的记忆
                        related_agents = self.extract_agents_from_text(node.describe)
                        # print(f"  [extract_cognitive_graph] 从事件记忆提取到的 agents: {related_agents}") # 核心：显示从该记忆中提取的agent
                        event_agents_found.update(related_agents)
                        for related_agent in related_agents:
                            if related_agent != agent.name:
                                G.add_node(related_agent)
                                if G.has_edge(agent.name, related_agent):
                                    G[agent.name][related_agent]['weight'] += 0.5
                                    G[agent.name][related_agent]['event_count'] += 1
                                else:
                                    G.add_edge(agent.name, related_agent,
                                               weight=0.5, chat_count=0, event_count=1, type='event')
            except Exception as e:
                print(f"检索 {agent.name} 的事件记忆时出错 (查询: {query}): {e}")
                continue

        all_agents_found = chat_agents_found.union(event_agents_found)
        print(f"--- {agent.name} 认知图提取统计 ---") # 核心：Agent提取总结
        print(f"  对话记忆检索数量: {chat_memories_count}")
        print(f"  事件记忆检索数量: {event_memories_count}")
        print(f"  从对话记忆中提取到的Agent: {chat_agents_found if chat_agents_found else '无'}")
        print(f"  从事件记忆中提取到的Agent: {event_agents_found if event_agents_found else '无'}")
        print(f"  总共提取到的相关Agent: {all_agents_found if all_agents_found else '无'}")
        print(f"  最终认知图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        return G

    def _initialize_agent_memory(self, agent):
        """初始化Agent的记忆字典，从docstore中获取所有节点ID并按类型分类"""
        try:
            # print(f"开始初始化 {agent.name} 的记忆...") # 状态信息
            if not hasattr(agent, 'associate') or not hasattr(agent.associate, '_index'):
                print(f"错误: Agent {agent.name} 缺少 associate 或 _index 属性")
                return

            index = agent.associate._index
            all_nodes = []

            if hasattr(index, '_index') and hasattr(index._index, 'docstore'):
                docstore = index._index.docstore
                if hasattr(docstore, 'docs') and docstore.docs:
                    all_nodes = list(docstore.docs.values())
                    # print(f"  从docstore.docs中获取到 {len(all_nodes)} 个节点") # 核心：节点加载信息
            
            if not all_nodes and hasattr(index, 'get_nodes'): # 备用方案
                try:
                    all_nodes = index.get_nodes()
                    # print(f"  从get_nodes()中获取到 {len(all_nodes)} 个节点") # 核心：节点加载信息
                except Exception as e:
                    print(f"  get_nodes()调用失败: {e}")

            if hasattr(index, '_path') and index._path:
                docstore_path = os.path.join(index._path, 'docstore.json')
                if os.path.exists(docstore_path):
                    try:
                        with open(docstore_path, 'r', encoding='utf-8') as f:
                            docstore_data = json.load(f)
                            if 'docstore/data' in docstore_data:
                                stored_nodes_count = len(docstore_data['docstore/data'])
                                # print(f"  文件 {docstore_path} 中存储的节点数: {stored_nodes_count}") # 核心：文件节点信息
                                if not all_nodes and stored_nodes_count > 0:
                                    print(f"  警告: 文件中有{stored_nodes_count}个节点，但内存中为空，可能存在加载问题")
                    except Exception as e:
                        print(f"  读取存储文件 {docstore_path} 失败: {e}")
                # else:
                    # print(f"  存储路径 {docstore_path} 不存在")

            # print(f"  最终为 {agent.name} 获取到 {len(all_nodes)} 个记忆节点") # 核心：最终节点数
            memory_by_type = {"event": [], "thought": [], "chat": []}
            for node in all_nodes:
                node_type = None
                if hasattr(node, 'metadata') and 'node_type' in node.metadata:
                    node_type = node.metadata['node_type']
                elif hasattr(node, 'extra_info') and 'node_type' in node.extra_info:
                    node_type = node.extra_info['node_type']
                
                if node_type in memory_by_type:
                    memory_by_type[node_type].append(node.id_)

            agent.associate.memory = memory_by_type
            # print(f"  {agent.name} 记忆分类统计: 对话={len(memory_by_type['chat'])}, 事件={len(memory_by_type['event'])}, 思考={len(memory_by_type['thought'])}") # 核心：分类统计
        except Exception as e:
            print(f"初始化 {agent.name} 记忆字典时出错: {e}")
            # print(f"错误详情: {traceback.format_exc()}") # 调试时可取消注释
            if not hasattr(agent.associate, 'memory') or not agent.associate.memory:
                agent.associate.memory = {"event": [], "thought": [], "chat": []}


class RealWorldGraphExtractor:
    """真实世界图提取器 - 从对话数据中提取真实关系图"""

    def extract_participants(self, location_key: str) -> List[str]:
        """从位置键中提取参与者"""
        participants_part = location_key.split(" @ ")[0] if " @ " in location_key else location_key
        participants_raw = participants_part.split(" -> ")
        return [name.strip() for name in participants_raw if name.strip()]

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
                    for i, agent1 in enumerate(participants):
                        G.add_node(agent1)
                        for agent2 in participants[i+1:]:
                            G.add_node(agent2)
                            chat_rounds = len(chat_log) if isinstance(chat_log, list) else 1
                            if G.has_edge(agent1, agent2):
                                G[agent1][agent2]['weight'] += chat_rounds
                                G[agent1][agent2]['interaction_count'] += 1
                            else:
                                G.add_edge(agent1, agent2,
                                           weight=chat_rounds, interaction_count=1)
                except Exception as e:
                    print(f"处理对话组时出错 (key: {location_key}): {e}")
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
        intersection = cognitive_nodes.intersection(real_nodes)
        union = cognitive_nodes.union(real_nodes)
        jaccard_similarity = len(intersection) / len(union) if union else 0
        missing_nodes = real_nodes - cognitive_nodes
        hallucinated_nodes = cognitive_nodes - real_nodes
        return {
            'node_count_diff': abs(len(cognitive_nodes) - len(real_nodes)),
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
        intersection = cognitive_edges.intersection(real_edges)
        union = cognitive_edges.union(real_edges)
        edge_jaccard = len(intersection) / len(union) if union else 0
        missing_edges = real_edges - cognitive_edges
        hallucinated_edges = cognitive_edges - real_edges
        return {
            'edge_count_diff': abs(len(cognitive_edges) - len(real_edges)),
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
            max_w = max(cognitive_weight, real_weight)
            if max_w > 0:
                relative_diff = abs(cognitive_weight - real_weight) / max_w
                weight_differences.append(1 - relative_diff)
        return np.mean(weight_differences) if weight_differences else 0.0

    def analyze_memory_reality_consistency_with_llm(self, agent_memories: List[str],
                                                   real_conversations: List[str]) -> Dict:
        """使用LLM分析记忆与真实对话的一致性"""
        if not self.llm_model:
            # print("未提供LLM模型，无法进行语义一致性分析") # 状态信息
            return {'consistency_score': 0.0, 'analysis': '无LLM模型'}
        try:
            # print(f"开始LLM记忆一致性分析 (记忆: {len(agent_memories)}条, 对话: {len(real_conversations)}条)...") # 状态信息
            memories_text = "\n".join([f"记忆{i+1}: {mem[:200]}" for i, mem in enumerate(agent_memories[:3])]) # 减少样本量
            conversations_text = "\n".join([f"对话{i+1}: {conv[:200]}" for i, conv in enumerate(real_conversations[:3])]) # 减少样本量
            prompt = f"""请分析以下Agent的记忆内容与真实对话记录的一致性程度。

**Agent记忆内容：**
{memories_text if memories_text else "无相关记忆"}

**真实对话记录：**
{conversations_text if conversations_text else "无相关对话"}

请从事实、情感、时间、人物关系等维度进行分析，给出一个0-1之间的一致性评分（1表示完全一致），并简要说明理由。
输出格式：
一致性评分：[0.0-1.0]
分析理由：[简要说明]"""
            response = self.llm_model.completion(prompt, temperature=0.1)
            if not response:
                return {'consistency_score': 0.0, 'analysis': 'LLM响应为空'}
            consistency_score = self._extract_consistency_score(response)
            analysis = self._extract_analysis_reason(response)
            return {'consistency_score': consistency_score, 'analysis': analysis} # 'llm_response': response # 调试时可加回
        except Exception as e:
            print(f"LLM语义一致性分析时出错: {e}")
            return {'consistency_score': 0.0, 'analysis': f'分析出错: {str(e)}'}

    def _extract_consistency_score(self, llm_response: str) -> float:
        """从LLM响应中提取一致性评分"""
        try:
            score_patterns = [r'一致性评分[：:]\s*(\d*\.?\d+)', r'评分[：:]\s*(\d*\.?\d+)', r'(\d*\.?\d+)\s*分', r'(0\.[0-9]+|1\.0|1|0)']
            for pattern in score_patterns:
                match = re.search(pattern, llm_response)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
            # 文本推断 (简化)
            if any(kw in llm_response for kw in ['完全一致', '高度一致']): return 0.9
            if any(kw in llm_response for kw in ['基本一致', '大部分一致']): return 0.7
            return 0.5 # 默认值
        except Exception as e:
            print(f"提取一致性评分时出错: {e}")
            return 0.0

    def _extract_analysis_reason(self, llm_response: str) -> str:
        """从LLM响应中提取分析理由"""
        try:
            reason_patterns = [r'分析理由[：:](.*?)(?:\n\n|$)', r'理由[：:](.*?)(?:\n\n|$)', r'说明[：:](.*?)(?:\n\n|$)']
            for pattern in reason_patterns:
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    return match.group(1).strip()
            # 简化提取逻辑：取非评分后的第一段有效文本
            lines = llm_response.split('\n')
            for line in lines:
                l_strip = line.strip()
                if l_strip and '评分' not in l_strip and len(l_strip) > 10:
                    return l_strip[:250] # 限制长度
            return llm_response[:250] # 默认返回前缀
        except Exception as e:
            print(f"提取分析理由时出错: {e}")
            return "无法提取分析理由"

    def compare_interaction_patterns_with_llm(self, cognitive_interactions: Dict,
                                            real_interactions: Dict) -> Dict:
        """使用LLM比较交互模式的相似性"""
        if not self.llm_model:
            return {'pattern_similarity': 0.0, 'analysis': '无LLM模型'}
        try:
            # print(f"开始LLM交互模式比较 (认知交互: {len(cognitive_interactions)}条, 真实交互: {len(real_interactions)}条)...") # 状态信息
            cognitive_desc = self._describe_interaction_patterns(cognitive_interactions)
            real_desc = self._describe_interaction_patterns(real_interactions)
            prompt = f"""请比较以下两种交互模式的相似性：

**Agent认知中的交互模式：**
{cognitive_desc}

**真实的交互模式：**
{real_desc}

请分析交互频率、对象、内容类型、时机等方面的相似性，给出一个0-1之间的相似性评分（1表示完全相似），并简要说明理由。
输出格式：
相似性评分：[0.0-1.0]
分析理由：[简要说明]"""
            response = self.llm_model.completion(prompt, temperature=0.1)
            if not response:
                return {'pattern_similarity': 0.0, 'analysis': 'LLM响应为空'}
            similarity_score = self._extract_consistency_score(response)
            analysis = self._extract_analysis_reason(response)
            return {'pattern_similarity': similarity_score, 'analysis': analysis} # 'llm_response': response # 调试时可加回
        except Exception as e:
            print(f"LLM交互模式比较时出错: {e}")
            return {'pattern_similarity': 0.0, 'analysis': f'比较出错: {str(e)}'}

    def _extract_interaction_patterns(self, graph: nx.Graph) -> Dict:
        """从图中提取交互模式"""
        interactions = {}
        for u, v, data in graph.edges(data=True):
            # 确保节点顺序一致，便于比较
            pair = tuple(sorted((str(u), str(v))))
            agent_pair = f"{pair[0]}-{pair[1]}"
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
        # 限制描述数量，避免prompt过长
        descriptions = []
        for agent_pair, data in list(interactions.items())[:3]: # 限制为3条
            if isinstance(data, dict):
                desc = f"{agent_pair}: 权重{data.get('weight', 0):.1f}"
                if data.get('chat_count',0) > 0 : desc += f", 对话{data['chat_count']}次"
                if data.get('event_count',0) > 0 : desc += f", 事件{data['event_count']}次"
                descriptions.append(desc)
            else: # 兼容旧格式或错误数据
                descriptions.append(f"{agent_pair}: {str(data)[:50]}") # 限制长度
        return "\n".join(descriptions) if descriptions else "无有效交互数据"


    def calculate_temporal_consistency(self, cognitive_graph: nx.Graph, real_graph: nx.Graph) -> float:
        """计算时间一致性 (简化版本，基于图的密度)"""
        cognitive_density = nx.density(cognitive_graph) if cognitive_graph.number_of_nodes() > 1 else 0
        real_density = nx.density(real_graph) if real_graph.number_of_nodes() > 1 else 0
        if real_density == 0:
            return 1.0 if cognitive_density == 0 else 0.0
        density_similarity = 1 - abs(cognitive_density - real_density) / real_density
        return max(0, density_similarity)


class CognitiveWorldGapAnalyzer:
    """认知世界差距分析器 - 主要分析类"""

    def __init__(self, agents: Dict = None, conversation_data: Dict = None, llm_model=None, known_agents=None):
        self.agents = agents or {}
        self.conversation_data = conversation_data or {}
        self.llm_model = llm_model
        self.known_agents = list(set(known_agents)) if known_agents else list(agents.keys()) # 确保唯一性

        self.cognitive_extractor = CognitiveGraphExtractor(known_agents=self.known_agents, llm_model=llm_model)
        self.real_world_extractor = RealWorldGraphExtractor()
        self.diff_calculator = GraphDifferenceCalculator(llm_model=llm_model)
        self.real_world_graph = self.real_world_extractor.extract_real_world_graph(self.conversation_data)

    def load_conversation_data(self, filepath: str) -> bool:
        """加载对话数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_data = json.load(f)
            self.real_world_graph = self.real_world_extractor.extract_real_world_graph(self.conversation_data)
            return True
        except Exception as e:
            print(f"加载对话数据失败: {e}")
            return False

    def get_agent_real_subgraph(self, agent_name: str) -> nx.Graph:
        """获取与特定agent相关的真实世界子图 (仅包含该agent及其直接邻居)"""
        if agent_name not in self.real_world_graph:
            return nx.Graph() # 返回空图
        
        # 获取该agent的邻居节点
        neighbors = list(self.real_world_graph.neighbors(agent_name))
        subgraph_nodes = [agent_name] + neighbors
        
        # 创建子图，只包含这些节点以及它们之间的边
        return self.real_world_graph.subgraph(subgraph_nodes).copy()


    def cognitive_world_gap_metric(self, cognitive_graph: nx.Graph, real_graph: nx.Graph,
                                 weights: Dict = None) -> Dict:
        """综合认知差距度量"""
        weights = weights or {'node_structure': 0.3, 'edge_structure': 0.4,
                               'semantic_accuracy': 0.2, 'temporal_consistency': 0.1}
        node_metrics = self.diff_calculator.node_difference_metric(cognitive_graph, real_graph)
        edge_metrics = self.diff_calculator.edge_difference_metric(cognitive_graph, real_graph)
        node_structure_score = node_metrics['jaccard_similarity']
        edge_structure_score = edge_metrics['edge_jaccard_similarity']
        semantic_accuracy = self.diff_calculator.calculate_semantic_accuracy(cognitive_graph, real_graph)
        temporal_consistency = self.diff_calculator.calculate_temporal_consistency(cognitive_graph, real_graph)
        overall_score = (weights['node_structure'] * node_structure_score +
                         weights['edge_structure'] * edge_structure_score +
                         weights['semantic_accuracy'] * semantic_accuracy +
                         weights['temporal_consistency'] * temporal_consistency)
        return {
            'cognitive_gap': 1 - overall_score, # 核心：认知差距
            'overall_score': overall_score,
            'node_structure_score': node_structure_score,
            'edge_structure_score': edge_structure_score,
            'semantic_accuracy': semantic_accuracy,
            'temporal_consistency': temporal_consistency,
            'detailed_metrics': {'nodes': node_metrics, 'edges': edge_metrics}
        }

    def analyze_agent_cognitive_gap(self, agent_name: str) -> Dict:
        """分析单个agent的认知差距"""
        if agent_name not in self.agents:
            print(f"错误: Agent {agent_name} 未在分析器中找到") # 核心错误
            return {'error': f'Agent {agent_name} not found'}

        agent = self.agents[agent_name]
        print(f"\n--- 开始分析Agent: {agent_name} ---") # 核心：分析开始标志

        cognitive_graph = self.cognitive_extractor.extract_cognitive_graph(agent)
        real_subgraph = self.get_agent_real_subgraph(agent_name)
        
        if not real_subgraph.nodes():
            print(f"警告: Agent {agent_name} 在真实世界图中没有邻居或不存在，将使用空图进行比较。")


        gap_metrics = self.cognitive_world_gap_metric(cognitive_graph, real_subgraph)
        llm_analysis = {}

        if self.llm_model:
            try:
                # 提取有限的记忆和对话用于LLM分析，避免过载
                agent_memories = []
                # 简化记忆提取逻辑，假设agent.associate.memory已由_initialize_agent_memory填充
                if hasattr(agent, 'associate') and hasattr(agent.associate, 'memory'):
                    all_memory_ids = agent.associate.memory.get('chat', []) + agent.associate.memory.get('event', [])
                    # 随机抽取或按重要性排序后取前N个，这里简化为取前几个
                    # 实际应通过agent.associate._retrieve_by_ids等方法获取内容
                    # 此处仅为示例，真实场景需要从docstore获取node.describe
                    # agent_memories = [f"示例记忆 {i}" for i in range(min(5, len(all_memory_ids)))] # 占位符

                real_conversations_for_llm = []
                if self.conversation_data:
                    count = 0
                    for _, chats_at_time in self.conversation_data.items():
                        if count >= 3: break # 限制对话数量
                        if not isinstance(chats_at_time, list): continue
                        for chat_group in chats_at_time:
                            if count >= 3: break
                            if not isinstance(chat_group, dict) or len(chat_group.keys()) != 1: continue
                            location_key = list(chat_group.keys())[0]
                            chat_log = chat_group[location_key]
                            participants = self.real_world_extractor.extract_participants(location_key)
                            if agent_name in participants and isinstance(chat_log, list):
                                conv_text = "".join([f"{entry[0]}: {entry[1]}\n" for entry in chat_log if isinstance(entry, list) and len(entry) >=2])
                                if conv_text.strip():
                                    real_conversations_for_llm.append(conv_text.strip())
                                    count +=1
                
                # print(f"  为 {agent_name} 提取到 {len(agent_memories)} 条记忆和 {len(real_conversations_for_llm)} 段对话用于LLM分析。")


                memory_consistency = self.diff_calculator.analyze_memory_reality_consistency_with_llm(
                    agent_memories, real_conversations_for_llm # 使用筛选后的数据
                )
                cognitive_interactions = self.diff_calculator._extract_interaction_patterns(cognitive_graph)
                real_interactions = self.diff_calculator._extract_interaction_patterns(real_subgraph)
                interaction_similarity = self.diff_calculator.compare_interaction_patterns_with_llm(
                    cognitive_interactions, real_interactions
                )
                llm_analysis = {'memory_consistency': memory_consistency, 'interaction_similarity': interaction_similarity}
                print(f"  LLM分析完成 - Agent: {agent_name}") # 核心：LLM分析完成
            except Exception as e:
                print(f"  LLM分析失败 - Agent: {agent_name}, 错误: {str(e)}") # 核心错误
                llm_analysis = {'error': str(e)}
        
        print(f"--- Agent: {agent_name} 分析结束 ---") # 核心：分析结束标志
        return {
            'agent_name': agent_name,
            'cognitive_nodes': len(cognitive_graph.nodes()),
            'cognitive_edges': len(cognitive_graph.edges()),
            'real_nodes': len(real_subgraph.nodes()),
            'real_edges': len(real_subgraph.edges()),
            'gap_metrics': gap_metrics,
            'llm_analysis': llm_analysis,
            # 'cognitive_graph_info': {'nodes': list(cognitive_graph.nodes()), 'edges': [(u,v,d) for u,v,d in cognitive_graph.edges(data=True)]}, # 可选：详细图信息
            # 'real_graph_info': {'nodes': list(real_subgraph.nodes()), 'edges': [(u,v,d) for u,v,d in real_subgraph.edges(data=True)]}  # 可选：详细图信息
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
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        if not valid_results:
            print("错误: 没有有效的agent分析结果") # 核心错误
            return {'error': 'No valid agent analysis results', 'total_agents': len(self.agents), 'analyzed_agents': 0}

        cognitive_gaps = [r['gap_metrics']['cognitive_gap'] for r in valid_results.values() if 'gap_metrics' in r and 'cognitive_gap' in r['gap_metrics']]
        if not cognitive_gaps: # 如果没有有效的gap数据
             return {'error': 'No valid cognitive gap data for summary', 'detailed_results': valid_results}


        max_gap_item = max(valid_results.items(), key=lambda x: x[1].get('gap_metrics', {}).get('cognitive_gap', -1))
        min_gap_item = min(valid_results.items(), key=lambda x: x[1].get('gap_metrics', {}).get('cognitive_gap', 2))


        report = {
            'summary': {
                'total_agents': len(self.agents),
                'analyzed_agents': len(valid_results),
                'average_cognitive_gap': np.mean(cognitive_gaps) if cognitive_gaps else -1,
                'std_cognitive_gap': np.std(cognitive_gaps) if cognitive_gaps else -1,
                'max_gap_agent': {'name': max_gap_item[0], 'gap': max_gap_item[1]['gap_metrics']['cognitive_gap']},
                'min_gap_agent': {'name': min_gap_item[0], 'gap': min_gap_item[1]['gap_metrics']['cognitive_gap']},
                'real_world_graph_info': {
                    'total_nodes': len(self.real_world_graph.nodes()),
                    'total_edges': len(self.real_world_graph.edges()),
                    'density': nx.density(self.real_world_graph) if self.real_world_graph.nodes() else 0
                }
            },
            'detailed_results': valid_results
        }
        return report

    def save_report(self, report: Dict, filepath: str) -> bool:
        """保存报告到文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: '<not serializable>') # 处理无法序列化的对象
            return True
        except Exception as e:
            print(f"保存报告失败: {e}") # 核心错误
            return False

    def print_summary(self, report: Dict = None):
        """打印报告摘要"""
        if report is None:
            report = self.generate_gap_report()
        if 'error' in report and report['error'] == 'No valid agent analysis results': # 特殊处理无结果的情况
            print(f"错误: {report['error']}. 总Agent数量: {report.get('total_agents', '未知')}, 成功分析: {report.get('analyzed_agents',0)}")
            return
        if 'summary' not in report:
            print(f"错误: 报告中缺少 'summary' 部分。报告内容: {str(report)[:200]}...")
            return


        summary = report['summary']
        print("\n=== Agent认知差距分析报告 ===") # 核心：报告标题
        print(f"总Agent数量: {summary['total_agents']}")
        print(f"成功分析的Agent数量: {summary['analyzed_agents']}")
        if summary.get('average_cognitive_gap', -1) != -1:
            print(f"平均认知差距: {summary['average_cognitive_gap']:.3f}")
            print(f"认知差距标准差: {summary['std_cognitive_gap']:.3f}")
            print(f"认知差距最大的Agent: {summary['max_gap_agent']['name']} (差距: {summary['max_gap_agent']['gap']:.3f})")
            print(f"认知差距最小的Agent: {summary['min_gap_agent']['name']} (差距: {summary['min_gap_agent']['gap']:.3f})")
        else:
            print("未能计算认知差距统计数据。")

        print(f"\n真实世界图信息:")
        print(f"  节点数: {summary['real_world_graph_info']['total_nodes']}")
        print(f"  边数: {summary['real_world_graph_info']['total_edges']}")
        print(f"  密度: {summary['real_world_graph_info']['density']:.3f}")
        print("\n--- 各Agent详细分析摘要 ---") # 核心：详细分析标题
        for agent_name, result in report.get('detailed_results', {}).items():
            if 'gap_metrics' in result and 'cognitive_gap' in result['gap_metrics']:
                 gap = result['gap_metrics']['cognitive_gap']
                 print(f"{agent_name}: 认知差距={gap:.3f}, "
                      f"认知图({result.get('cognitive_nodes','N/A')}节点,{result.get('cognitive_edges','N/A')}边), "
                      f"真实图({result.get('real_nodes','N/A')}节点,{result.get('real_edges','N/A')}边)")
            else:
                print(f"{agent_name}: 未能计算认知差距。")


# --- Helper Functions and Main Execution ---
# (以下函数中的打印信息多为用户交互或状态指示，予以保留或稍作调整)

def load_agent_from_storage(agent_name: str, storage_path: str, config: Dict = None) -> Optional[Any]: # 'Agent' 类型提示改为 Any
    """从存储路径加载agent (简化版，用于分析)"""
    try:
        # 动态导入Agent类，假设其在 'modules.agent'
        # 实际项目中，需要确保PYTHONPATH正确设置或使用相对导入
        import sys
        # 假设脚本在项目根目录的某个子目录下执行，需要调整路径以找到 'modules'
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # 示例路径调整
        # if project_root not in sys.path:
        #    sys.path.insert(0, project_root)
        from modules.agent import Agent # 尝试导入

        # 简化配置加载，实际应与Agent的初始化方式匹配
        agent_config_path = Path(storage_path).parent.parent / "agents" / agent_name / "agent.json" # 推断原始配置文件路径
        if not agent_config_path.exists(): # 退回到使用传入的storage_path下的agent.json
             agent_config_path = Path(storage_path) / "agent.json"

        if not agent_config_path.exists():
            print(f"警告: 找不到agent配置文件: {agent_config_path}，将使用默认或传入的config")
            loaded_config = config or {}
        else:
            try:
                with open(agent_config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                if config: # 合并传入的config，传入的优先
                    loaded_config.update(config)
            except Exception as e:
                print(f"读取agent配置文件 {agent_config_path} 失败: {e}，将使用默认或传入的config")
                loaded_config = config or {}
        
        # 确保核心配置存在
        final_config = {
            "name": loaded_config.get("name", agent_name),
            "storage_root": str(Path(storage_path).parent), # 指向 <agent_name> 的父目录，即 .../storage/
            "percept": loaded_config.get("percept", {"vision_r": 3, "att_bandwidth": 3}),
            "think": loaded_config.get("think", {"interval": 10}),
            "associate": loaded_config.get("associate", { # 确保embedding配置存在
                "embedding": {"type": "openai", "model": "text-embedding-3-small"},
                "retention": 8
            }),
            "llm": loaded_config.get("llm", {"type": "openai", "model": "gpt-4o-mini"}),
            # 其他必要配置...
        }
        final_config.update(loaded_config) # 把原始配置文件的其他项也加上

        # 模拟环境依赖 (Maze, Conversation, Logger)
        class DummyMaze:
            def tile_at(self, coord): return type('DummyTile', (), {'get_address': lambda: "dummy_address", 'events': {}})()
            def get_scope(self, coord, config): return []
        class DummyConversation: pass
        class DummyLogger: info = print

        agent_instance = Agent(final_config, DummyMaze(), DummyConversation(), DummyLogger())
        # print(f"Agent {agent_name} 存储路径: {agent_instance.storage_path}") # 确认存储路径
        return agent_instance
    except ImportError:
        print("错误: 无法导入 'modules.agent.Agent'。请确保模块路径正确。")
        return None
    except Exception as e:
        print(f"加载agent {agent_name} 失败: {e}")
        # print(traceback.format_exc()) # 调试时可取消注释
        return None


def select_simulation() -> Optional[str]:
    """让用户选择一个模拟结果进行分析"""
    checkpoints_dir = "results/checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"错误: 找不到checkpoints目录: {checkpoints_dir}")
        return None
    simulations = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and os.path.exists(os.path.join(checkpoints_dir, d, "conversation.json"))]
    if not simulations:
        print("错误: 没有找到任何有效的模拟 (包含conversation.json)")
        return None
    print("\n=== 可用的模拟 ===")
    for i, sim in enumerate(simulations, 1): print(f"{i}. {sim}")
    while True:
        try:
            choice = input(f"请选择模拟 (1-{len(simulations)}) 或输入 'q' 退出: ").strip()
            if choice.lower() == 'q': return None
            idx = int(choice) - 1
            if 0 <= idx < len(simulations): return simulations[idx]
            print(f"无效选择，请输入 1-{len(simulations)} 之间的数字")
        except ValueError: print("无效输入，请输入数字")


def select_agents(simulation_name: str) -> Optional[List[str]]:
    """让用户选择要分析的agents"""
    storage_dir = f"results/checkpoints/{simulation_name}/storage"
    if not os.path.exists(storage_dir):
        print(f"错误: 找不到storage目录: {storage_dir}")
        return None
    agents = [d for d in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, d))]
    if not agents:
        print(f"错误: 在 {simulation_name} 中没有找到任何agent")
        return None
    print(f"\n=== {simulation_name} 中的Agents ===")
    for i, agent in enumerate(agents, 1): print(f"{i}. {agent}")
    print(f"{len(agents) + 1}. 全部分析")
    while True:
        try:
            choice_str = input(f"选择要分析的agent (1-{len(agents) + 1}, 可多选逗号隔开) 或输入 'q' 退出: ").strip()
            if choice_str.lower() == 'q': return None
            
            choices = []
            parts = choice_str.split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                idx = int(part) - 1
                if idx == len(agents): # 全部分析
                    return agents 
                if 0 <= idx < len(agents):
                    choices.append(agents[idx])
                else:
                    print(f"无效选择: {part}。请输入 1-{len(agents) + 1} 之间的数字")
                    choices = [] # 重置选择
                    break 
            if choices:
                return list(set(choices)) #去重后返回

        except ValueError: print("无效输入，请输入数字或逗号分隔的数字")


def load_llm_config_from_project():
    """从项目标准位置加载LLM配置"""
    # 假设配置文件在项目根目录下的 data/config.json
    # config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "config.json") # 示例路径
    config_path = "data/config.json" # 简化路径，假设脚本在特定位置运行
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # 提取LLM相关配置，路径可能需要根据实际config.json结构调整
        llm_config = config.get("agent", {}).get("think", {}).get("llm", {})
        llm_config["keys"] = config.get("api_keys", {}) # API密钥通常在顶层
        if not llm_config.get("model"): # 如果think中没有，尝试顶层llm配置
             llm_config = config.get("llm", llm_config) # 使用已有的llm_config作为默认值
             llm_config["keys"] = config.get("api_keys", {})

        if not llm_config.get("model") or not llm_config.get("base_url"):
            print("警告: LLM配置不完整 (缺少model或base_url)")
            return None
        return llm_config
    except FileNotFoundError:
        print(f"警告：找不到LLM配置文件 {config_path}")
        return None
    except Exception as e:
        print(f"警告：无法加载LLM配置文件 {config_path}：{e}")
        return None


def create_llm_instance_from_project_config():
    """根据项目配置创建LLM实例"""
    try:
        from modules.model.llm_model import create_llm_model # 假设的LLM工厂函数
        llm_config = load_llm_config_from_project()
        if not llm_config:
            print("无法加载LLM配置，无法创建LLM实例。")
            return None
        
        print("正在创建 LLM 实例...") # 状态信息
        llm_instance = create_llm_model(
            base_url=llm_config.get("base_url"),
            model=llm_config.get("model"),
            embedding_model=llm_config.get("embedding",{}).get("model"), # 兼容旧的embedding位置
            keys=llm_config.get("keys", {})
        )
        if llm_instance and hasattr(llm_instance, 'is_available') and llm_instance.is_available(): # 假设有is_available方法
            print(f"✓ LLM 实例 ({llm_instance.__class__.__name__}) 创建成功") # 核心：成功信息
            return llm_instance
        else:
            print("LLM 实例创建失败或不可用。") # 核心：失败信息
            return None
    except ImportError:
        print("错误: 无法导入 'modules.model.llm_model'。请确保LLM模块路径正确。")
        return None
    except Exception as e:
        print(f"创建 LLM 实例时出错：{e}")
        return None


def main():
    """主函数 - 交互式认知差距分析"""
    print("\n=== Agent认知模型与真实世界差距度量分析器 ===") # 核心：程序标题

    simulation_name = select_simulation()
    if not simulation_name: print("退出程序。"); return
    print(f"\n选择的模拟: {simulation_name}") # 核心：用户选择

    selected_agent_names = select_agents(simulation_name)
    if not selected_agent_names: print("退出程序。"); return
    print(f"选择的agents: {', '.join(selected_agent_names)}") # 核心：用户选择

    print("\n--- LLM 初始化 ---") # 核心：阶段标志
    llm_model = create_llm_instance_from_project_config()
    if llm_model: print("✓ LLM 模型初始化成功，将启用 LLM 增强分析。") # 核心：状态
    else: print("⚠ LLM 模型初始化失败，将使用传统分析方法。") # 核心：状态
    print("--- LLM 初始化结束 ---\n") # 核心：阶段标志

    print("正在加载agents...") # 核心：状态
    loaded_agents_map = {}
    for agent_name in selected_agent_names:
        # storage_path应指向agent的具体数据目录，例如 .../storage/<agent_name>/
        storage_path_for_agent = f"results/checkpoints/{simulation_name}/storage/{agent_name}"
        agent_instance = load_agent_from_storage(agent_name, storage_path_for_agent)
        if agent_instance:
            loaded_agents_map[agent_name] = agent_instance
            print(f"✓ 成功加载 {agent_name}") # 核心：成功信息
        else:
            print(f"✗ 加载 {agent_name} 失败") # 核心：失败信息
    if not loaded_agents_map: print("没有成功加载任何agent，退出程序。"); return

    conversation_path = f"results/checkpoints/{simulation_name}/conversation.json"
    
    # 获取所有可能的Agent名称作为 known_agents，以便更准确地从记忆中提取交互对象
    all_possible_agent_names_in_sim = []
    full_storage_dir = f"results/checkpoints/{simulation_name}/storage"
    if os.path.exists(full_storage_dir):
        all_possible_agent_names_in_sim = [d for d in os.listdir(full_storage_dir) if os.path.isdir(os.path.join(full_storage_dir, d))]
    
    analyzer = CognitiveWorldGapAnalyzer(agents=loaded_agents_map, llm_model=llm_model, known_agents=all_possible_agent_names_in_sim)
    if not analyzer.load_conversation_data(conversation_path):
        print(f"加载对话数据失败: {conversation_path}，退出程序。") # 核心：失败信息
        return
    print(f"成功加载对话数据: {conversation_path}") # 核心：成功信息

    print("\n正在生成认知差距分析报告...") # 核心：状态
    report = analyzer.generate_gap_report()
    analyzer.print_summary(report) # 核心：打印摘要

    report_filename = f"cognitive_gap_report_{simulation_name}.json"
    report_dir = Path("results/analysis_reports") # 指定报告保存目录
    report_dir.mkdir(parents=True, exist_ok=True) # 创建目录
    report_path = report_dir / report_filename

    if analyzer.save_report(report, str(report_path)):
        print(f"\n报告已保存到: {report_path}") # 核心：成功信息
    else:
        print("\n保存报告失败。") # 核心：失败信息
    print("\n分析完成！") # 核心：结束标志


if __name__ == "__main__":
    # 为了能正确导入项目模块 (如 modules.agent)，可能需要调整 sys.path
    # 例如，如果此脚本在项目的 tools/ 子目录下：
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_dir, "..")) #假设项目根目录是上一级
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    main()
