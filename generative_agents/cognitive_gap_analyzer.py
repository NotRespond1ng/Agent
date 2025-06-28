# -*- coding: utf-8 -*-
"""
Agent认知模型与真实世界差距度量分析器

这是一个用于分析Agent认知模型与真实世界之间差距的工具。
主要功能包括：
1. 从Agent记忆中提取认知关系图
2. 从对话数据中提取真实关系图
3. 计算两个图之间的各种差异度量
4. 生成详细的分析报告

使用方法：
    python cognitive_gap_analyzer.py
"""

import os
import json
from typing import Dict, List, Optional

# 导入分析器模块
from modules.analyzer import (
    CognitiveGraphExtractor,
    RealWorldGraphExtractor, 
    GraphDifferenceCalculator,
    CognitiveWorldGapAnalyzer,
    load_agent_from_storage
)


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
            choice = input(f"\n选择要分析的agent (1-{len(agents) + 1}) 或输入 'q' 退出: ").strip()
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


def load_llm_config():
    """加载LLM配置"""
    config_path = os.path.join("data", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            llm_config = config["agent"]["think"]["llm"]
            llm_config["keys"] = config["api_keys"]
            return llm_config
    except Exception as e:
        print(f"警告：无法加载配置文件：{e}")
        return None


def create_llm_instance():
    """创建LLM实例"""
    try:
        from modules.model.llm_model import create_llm_model
        
        # 加载配置
        llm_config = load_llm_config()
        if not llm_config:
            print("无法加载LLM配置")
            return None
        
        # 创建LLM实例
        print("正在创建 LLM 实例...")
        llm_instance = create_llm_model(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            embedding_model=llm_config.get("embedding_model"),
            keys=llm_config["keys"]
        )
        
        if llm_instance and llm_instance.is_available():
            print(f"✓ LLM 实例 ({llm_instance.__class__.__name__}) 创建成功")
            return llm_instance
        else:
            print("LLM 实例创建失败或不可用")
            return None
            
    except ImportError as e:
        print(f"无法导入 LLM 模块：{e}")
        return None
    except Exception as e:
        print(f"创建 LLM 实例时出错：{e}")
        return None


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
    
    # 3. 创建LLM实例
    print("\n--- 开始 LLM 初始化 ---")
    llm_model = create_llm_instance()
    if llm_model:
        print("✓ LLM 模型初始化成功，将启用 LLM 增强分析")
    else:
        print("⚠ LLM 模型初始化失败，将使用传统分析方法")
    print("--- LLM 初始化结束 ---\n")
    
    # 4. 加载agents
    print("正在加载agents...")
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
    
    # 5. 加载对话数据并创建分析器
    conversation_path = f"results/checkpoints/{simulation_name}/conversation.json"
    
    # 获取所有可能的Agent名称作为 known_agents
    storage_dir = f"results/checkpoints/{simulation_name}/storage"
    all_agents = []
    if os.path.exists(storage_dir):
        for item in os.listdir(storage_dir):
            agent_path = os.path.join(storage_dir, item)
            if os.path.isdir(agent_path):
                all_agents.append(item)
    
    # 传递所有Agent名称作为known_agents参数，而不是只传递selected_agents
    analyzer = CognitiveWorldGapAnalyzer(agents=loaded_agents, llm_model=llm_model, known_agents=all_agents)
    
    if not analyzer.load_conversation_data(conversation_path):
        print(f"加载对话数据失败: {conversation_path}")
        return
    
    print(f"成功加载对话数据: {conversation_path}")
    
    if llm_model:
        print("✓ LLM增强分析已启用")
    else:
        print("⚠ 未找到LLM模型，将使用传统分析方法")
    
    # 6. 生成分析报告
    report = analyzer.generate_gap_report()
    
    # 7. 显示结果
    analyzer.print_summary(report)
    
    # 8. 保存报告
    report_path = f"results/checkpoints/{simulation_name}/cognitive_gap_report.json"
    if analyzer.save_report(report, report_path):
        print(f"\n报告已保存到: {report_path}")
    else:
        print("\n保存报告失败")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
