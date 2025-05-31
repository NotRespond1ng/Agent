# -*- coding: utf-8 -*-
"""
认知差距分析器 - 主要分析类
"""

import json
import os
from typing import Dict, List, Optional
import networkx as nx
import numpy as np

from .cognitive_graph_extractor import CognitiveGraphExtractor
from .real_world_graph_extractor import RealWorldGraphExtractor
from .graph_difference_calculator import GraphDifferenceCalculator


class CognitiveWorldGapAnalyzer:
    """认知世界差距分析器 - 主要分析类"""
    
    def __init__(self, agents: Dict = None, conversation_data: Dict = None, llm_model=None, known_agents=None):
        self.agents = agents or {}
        self.conversation_data = conversation_data or {}
        self.llm_model = llm_model
        
        # 如果没有提供known_agents，从agents字典中提取agent名称
        if known_agents is None and agents:
            known_agents = list(agents.keys())
        self.known_agents = known_agents
        
        self.cognitive_extractor = CognitiveGraphExtractor(known_agents=known_agents, llm_model=llm_model)
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
                # 提取agent的记忆内容
                agent_memories = []
                if hasattr(agent, 'memory') and agent.memory:
                    # 获取associative记忆
                    if hasattr(agent.memory, 'associative') and agent.memory.associative:
                        memories = agent.memory.associative.retrieve_relevant_memories(
                            "对话 交流 聊天", k=10
                        )
                        agent_memories.extend([mem.description for mem in memories])
                    
                    # 获取event记忆
                    if hasattr(agent.memory, 'event') and agent.memory.event:
                        events = agent.memory.event.retrieve_relevant_memories(
                            "对话 交流 聊天", k=10
                        )
                        agent_memories.extend([event.description for event in events])
                
                # 提取真实对话数据
                real_conversations = []
                if self.conversation_data:
                    for timestamp, chats_at_time in self.conversation_data.items():
                        if not isinstance(chats_at_time, list):
                            continue
                        
                        for chat_group in chats_at_time:
                            if not isinstance(chat_group, dict) or len(chat_group.keys()) != 1:
                                continue
                            
                            persons_location_key = list(chat_group.keys())[0]
                            chat_log = chat_group[persons_location_key]
                            
                            # 解析参与者
                            if " @ " in persons_location_key:
                                participants_part = persons_location_key.split(" @ ")[0]
                            else:
                                participants_part = persons_location_key
                            participants_raw = participants_part.split(" -> ")
                            
                            # 检查当前agent是否参与了这个对话
                            agent_participated = False
                            for participant in participants_raw:
                                if agent_name in participant:
                                    agent_participated = True
                                    break
                            
                            if agent_participated and isinstance(chat_log, list):
                                # 将对话记录转换为文本
                                conversation_text = ""
                                for chat_entry in chat_log:
                                    if isinstance(chat_entry, list) and len(chat_entry) >= 2:
                                        speaker, text = chat_entry[0], chat_entry[1]
                                        conversation_text += f"{speaker}: {text}\n"
                                
                                if conversation_text.strip():
                                    real_conversations.append(conversation_text.strip())
                
                memory_consistency = self.diff_calculator.analyze_memory_reality_consistency_with_llm(
                    agent_memories, real_conversations
                )
                
                # 分析交互模式的相似性
                # 从图中提取交互模式
                cognitive_interactions = self.diff_calculator._extract_interaction_patterns(cognitive_graph)
                real_interactions = self.diff_calculator._extract_interaction_patterns(real_subgraph)
                
                interaction_similarity = self.diff_calculator.compare_interaction_patterns_with_llm(
                    cognitive_interactions, real_interactions
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
        
        # 创建agent实例
        agent = Agent(config, DummyMaze(), DummyConversation(), DummyLogger())
        return agent
    except Exception as e:
        import traceback
        print(f"加载agent {agent_name} 失败: {e}")
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        return None


class DummyTile:
    """虚拟Tile类，用于Agent加载时的占位"""
    
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
    """虚拟Maze类，用于Agent加载时的占位"""
    
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
    """虚拟Conversation类，用于Agent加载时的占位"""
    pass


class DummyLogger:
    """虚拟Logger类，用于Agent加载时的占位"""
    
    def info(self, msg):
        pass