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

import networkx as nx
import numpy as np
from pathlib import Path


class CognitiveGraphExtractor:
    """认知图提取器 - 从agent记忆中提取认知关系图"""
    
    def __init__(self):
        self.agent_name_pattern = re.compile(r'[\u4e00-\u9fff]+|[A-Za-z]+')  # 匹配中文名或英文名
    
    def extract_agents_from_text(self, text: str) -> Set[str]:
        """从文本中提取agent名称"""
        # 常见的agent名称模式
        potential_names = self.agent_name_pattern.findall(text)
        
        # 过滤掉常见的非名称词汇
        common_words = {'正在', '此时', '对话', '聊天', '说话', '交谈', '讨论', '谈论', 
                       'is', 'was', 'are', 'were', 'the', 'and', 'or', 'but', 'chat', 'talk'}
        
        agent_names = set()
        for name in potential_names:
            if len(name) >= 2 and name not in common_words:
                agent_names.add(name)
        
        return agent_names
    
    def extract_cognitive_graph(self, agent) -> nx.Graph:
        """从agent的记忆中提取认知图"""
        G = nx.Graph()
        G.add_node(agent.name)  # 添加自己作为中心节点
        
        # 从对话记忆中提取关系
        for chat_node_id in agent.associate.memory.get('chat', []):
            try:
                concept = agent.associate.find_concept(chat_node_id)
                other_agents = self.extract_agents_from_text(concept.describe)
                
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
                print(f"处理chat节点 {chat_node_id} 时出错: {e}")
                continue
        
        # 从事件记忆中提取关系
        for event_node_id in agent.associate.memory.get('event', []):
            try:
                concept = agent.associate.find_concept(event_node_id)
                related_agents = self.extract_agents_from_text(concept.describe)
                
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
                print(f"处理event节点 {event_node_id} 时出错: {e}")
                continue
        
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
    
    def __init__(self, agents: Dict = None, conversation_data: Dict = None):
        self.agents = agents or {}
        self.conversation_data = conversation_data or {}
        
        self.cognitive_extractor = CognitiveGraphExtractor()
        self.real_world_extractor = RealWorldGraphExtractor()
        self.diff_calculator = GraphDifferenceCalculator()
        
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
        
        return {
            'agent_name': agent_name,
            'cognitive_nodes': len(cognitive_graph.nodes()),
            'cognitive_edges': len(cognitive_graph.edges()),
            'real_nodes': len(real_subgraph.nodes()),
            'real_edges': len(real_subgraph.edges()),
            'gap_metrics': gap_metrics,
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
                    "path": storage_path + "/associate",
                    "embedding": "text-embedding-ada-002",
                    "retention": 8,
                    "max_memory": -1,
                    "max_importance": 10
                },
                "currently": agent_config.get("currently", f"{agent_name}正在思考"),
                "scratch": agent_config.get("scratch", {}),
                "coord": agent_config.get("coord", [0, 0])
            }
        
        # 创建虚拟的maze和conversation对象
        class DummyMaze:
            def get_address_tiles(self, address):
                return [(0, 0)]
        
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
    
    # 3. 加载对话数据
    conversation_path = f"results/checkpoints/{simulation_name}/conversation.json"
    analyzer = CognitiveWorldGapAnalyzer()
    
    if not analyzer.load_conversation_data(conversation_path):
        print(f"加载对话数据失败: {conversation_path}")
        return
    
    print(f"成功加载对话数据: {conversation_path}")
    
    # 4. 加载agents
    print("\n正在加载agents...")
    loaded_agents = {}
    
    for agent_name in selected_agents:
        storage_path = f"results/checkpoints/{simulation_name}/storage/{agent_name}"
        agent = load_agent_from_storage(agent_name, storage_path)
        
        if agent:
            loaded_agents[agent_name] = agent
            print(f"✓ 成功加载 {agent_name}")
        else:
            print(f"✗ 加载 {agent_name} 失败")
    
    if not loaded_agents:
        print("没有成功加载任何agent，退出程序")
        return
    
    analyzer.agents = loaded_agents
    
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