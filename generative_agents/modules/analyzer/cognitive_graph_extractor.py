# -*- coding: utf-8 -*-
"""
认知图提取器 - 从agent记忆中提取认知关系图
"""

import re
import traceback
from typing import Set
import networkx as nx


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
        
        # 如果有已知的agent列表，直接在文本中查找这些名称
        if self.known_agents:
            return self.extract_agents_from_known_list(decoded_text)
        
        # 如果有LLM模型，使用智能识别（但不推荐，浪费token）
        if self.llm_model:
            print("警告：正在使用LLM提取agent名称，这会消耗token。建议提供known_agents参数。")
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
            
            return agent_names
            
        except Exception as e:
            # 静默回退到正则表达式方法
            return self.extract_agents_with_regex(text)
    
    def extract_agents_from_known_list(self, text: str) -> Set[str]:
        """从已知agent列表中查找在文本中出现的agent名称（推荐方法）"""
        found_agents = set()
        for agent_name_to_find in self.known_agents:
            if agent_name_to_find in text:
                found_agents.add(agent_name_to_find)
        
        # 保留原有的边界检查逻辑作为备用
        for agent_name in self.known_agents:
            # 检查agent名称是否在文本中出现
            if agent_name in text:
                # 进一步验证：确保是作为独立词汇出现，而不是其他词的一部分
                # 使用简单的边界检查
                import re
                pattern = r'\b' + re.escape(agent_name) + r'\b'
                if re.search(pattern, text):
                    found_agents.add(agent_name)
                elif agent_name in text:  # 对于中文名称，边界检查可能不适用
                    found_agents.add(agent_name)
        
        # 提取完成
        return found_agents
    
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
        
        # 开始记忆分析
        
        # 首先初始化Associate的memory字典，从docstore中获取所有节点
        try:
            self._initialize_agent_memory(agent)
        except Exception as e:
            print(f"初始化Agent记忆时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")
        
        # 使用类级别的LLM模型实例进行分析
        if self.llm_model:
            # 使用LLM模型进行分析
            pass
        else:
            print(f"未提供LLM模型，将使用传统的正则表达式方法")
        
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
                chat_nodes = agent.associate._retrieve_nodes('chat', query)
                
                for node in chat_nodes:
                    # Concept对象使用describe属性而不是text属性
                    if hasattr(node, 'describe') and node.describe:
                        chat_memories_count += 1
                        
                        other_agents = self.extract_agents_from_text(node.describe)
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
                # 静默处理检索错误
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
                event_nodes = agent.associate._retrieve_nodes('event', query)
                
                for node in event_nodes:
                    # Concept对象使用describe属性而不是text属性
                    if hasattr(node, 'describe') and node.describe:
                        event_memories_count += 1
                        
                        related_agents = self.extract_agents_from_text(node.describe)
                        event_agents_found.update(related_agents)
                        
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
    
    def _initialize_agent_memory(self, agent):
        """初始化Agent的记忆字典，从docstore中获取所有节点ID并按类型分类"""
        try:
            # 检查agent的记忆结构
            if not hasattr(agent, 'associate'):
                print("错误: agent没有associate属性")
                return []
            
            print(f"agent.associate类型: {type(agent.associate)}")
            
            if not hasattr(agent.associate, '_index'):
                print("错误: agent.associate没有_index属性")
                return []
            
            print(f"agent.associate._index类型: {type(agent.associate._index)}")
            
            # 检查索引的节点数量
            index = agent.associate._index
            if hasattr(index, 'nodes_num'):
                print(f"索引报告的节点数量: {index.nodes_num}")
            
            # 尝试不同的方法获取所有节点
            all_nodes = []
            
            # 方法1：尝试从docstore直接获取
            if hasattr(index, '_index') and hasattr(index._index, 'docstore'):
                try:
                    docstore = index._index.docstore
                    if hasattr(docstore, 'docs') and docstore.docs:
                        all_nodes = list(docstore.docs.values())
                        print(f"✓ 从docstore.docs中获取到 {len(all_nodes)} 个节点")
                        
                        # 显示前几个节点的信息
                        for i, node in enumerate(list(all_nodes)[:3]):
                            print(f"  节点{i+1}: ID={node.id_}, 文本长度={len(node.text) if hasattr(node, 'text') else 'N/A'}")
                    else:
                        print("✗ docstore.docs为空或不存在")
                except Exception as e:
                    print(f"✗ 从docstore获取节点失败: {e}")
            
            # 方法2：如果方法1失败，尝试get_nodes方法
            if not all_nodes and hasattr(index, 'get_nodes'):
                try:
                    all_nodes = index.get_nodes()
                    print(f"✓ 从get_nodes()中获取到 {len(all_nodes)} 个节点")
                except Exception as e:
                    print(f"✗ get_nodes()调用失败: {e}")
            
            # 方法3：检查存储路径
            if hasattr(index, '_path') and index._path:
                import os
                import json
                docstore_path = os.path.join(index._path, 'docstore.json')
                if os.path.exists(docstore_path):
                    print(f"✓ 存储路径存在: {docstore_path}")
                    try:
                        with open(docstore_path, 'r', encoding='utf-8') as f:
                            docstore_data = json.load(f)
                            if 'docstore/data' in docstore_data:
                                stored_nodes = len(docstore_data['docstore/data'])
                                print(f"  文件中存储的节点数: {stored_nodes}")
                                
                                # 如果内存中没有节点但文件中有，说明加载有问题
                                if not all_nodes and stored_nodes > 0:
                                    print(f"⚠️  警告: 文件中有{stored_nodes}个节点，但内存中为空，可能存在加载问题")
                    except Exception as e:
                        print(f"  读取存储文件失败: {e}")
                else:
                    print(f"✗ 存储路径不存在: {docstore_path}")
            
            print(f"最终获取到 {len(all_nodes)} 个节点")
            # 记忆初始化完成
            
            # 按类型分类节点ID
            memory_by_type = {"event": [], "thought": [], "chat": []}
            
            for node in all_nodes:
                if hasattr(node, 'metadata') and 'node_type' in node.metadata:
                    node_type = node.metadata['node_type']
                    if node_type in memory_by_type:
                        memory_by_type[node_type].append(node.id_)
                elif hasattr(node, 'extra_info') and 'node_type' in node.extra_info:
                    # 有些节点可能使用extra_info存储元数据
                    node_type = node.extra_info['node_type']
                    if node_type in memory_by_type:
                        memory_by_type[node_type].append(node.id_)
            
            # 更新Associate的memory字典
            agent.associate.memory = memory_by_type
            
            print(f"记忆分类统计:")
            # 记忆统计完成
            
        except Exception as e:
            print(f"初始化记忆字典时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            # 如果失败，至少确保memory字典存在
            if not hasattr(agent.associate, 'memory') or not agent.associate.memory:
                agent.associate.memory = {"event": [], "thought": [], "chat": []}