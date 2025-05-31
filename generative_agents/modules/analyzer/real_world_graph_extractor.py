# -*- coding: utf-8 -*-
"""
真实世界图提取器 - 从对话数据中提取真实关系图
"""

from typing import Dict, List
import networkx as nx


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