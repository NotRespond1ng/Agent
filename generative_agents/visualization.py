import json
import os
from collections import defaultdict
import networkx as nx
from pyvis.network import Network
import webbrowser
import sys
from pathlib import Path

# 获取当前文件的父目录的父目录（即 GenerativeAgentsCN/generative_agents 的路径）
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))  # 添加根目录到 Python 路径

import time

# --- 修改导入语句 ---
try:
    from modules.model.llm_model import create_llm_model, LLMModel
    LLM_AVAILABLE = True
    print("成功导入 LLM 模块。")
except ImportError as e:
    print(f"警告：无法导入 LLM 模块。将跳过对话总结功能。错误：{e}")
    LLM_AVAILABLE = False
    LLMModel = None


# --- LLM 配置：从项目配置文件加载 ---
def load_llm_config():
    config_path = os.path.join("", "data", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            llm_config = config["agent"]["think"]["llm"]
            llm_config["keys"] = config["api_keys"]
            return llm_config
    except Exception as e:
        print(f"警告：无法加载配置文件：{e}")
        return None


LLM_CONFIG = load_llm_config() or {
    "base_url": os.getenv("OPENAI_API_BASE", "YOUR_API_BASE_URL"),
    "model": os.getenv("LLM_MODEL_NAME", "YOUR_MODEL_NAME"),
    "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "YOUR_EMBEDDING_MODEL"),
    "keys": {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    },
    "enable_summarization": LLM_AVAILABLE and True
}

# --- 文件路径配置 ---
CONVERSATION_FILE_PATH = "results/checkpoints/test1/conversation.json"
OUTPUT_HTML_FILE = "./agent_relationship_network_with_summary.html"


# --- 数据加载 (保持不变) ---
def load_conversation_data(filepath):
    """从指定的JSON文件加载对话数据。"""
    if not os.path.exists(filepath):
        print(f"错误：在'{filepath}'未找到对话文件")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            conversation_data = json.load(f)
        print(f"成功从以下位置加载对话数据：{filepath}")
        return conversation_data
    except json.JSONDecodeError as e:
        print(f"错误：无法从'{filepath}'解码JSON。请检查文件格式。详情：{e}")
        return None
    except Exception as e:
        print(f"读取'{filepath}'时发生意外错误：{e}")
        return None


# --- 处理交互和收集对话 ---
def process_interactions_and_conversations(conversation_data):
    """处理对话数据以计算交互次数并收集对话文本。"""
    if conversation_data is None:
        return set(), defaultdict(int), defaultdict(list)

    interaction_counts = defaultdict(int)
    # 新增：存储每对agent之间的对话文本
    pairwise_conversations = defaultdict(list)
    agents = set()
    skipped_interactions = 0
    processed_interactions = 0

    print("正在处理对话数据并收集文本...")
    for timestamp, chats_at_time in conversation_data.items():
        if not isinstance(chats_at_time, list): continue  # 跳过格式不正确的时间戳条目

        for chat_group in chats_at_time:
            if not isinstance(chat_group, dict) or len(chat_group.keys()) != 1: continue  # 跳过格式不正确的对话组

            persons_location_key = list(chat_group.keys())[0]
            chat_log = chat_group[persons_location_key]  # 获取对话记录 [[speaker, text], ...]
            processed_interactions += 1

            try:
                # --- 解析参与者 (与之前相同) ---
                if " @ " in persons_location_key:
                    participants_part = persons_location_key.split(" @ ")[0]
                else:
                    participants_part = persons_location_key
                participants_raw = participants_part.split(" -> ")
                participants = [name.strip() for name in participants_raw if name.strip()]

                if not participants:
                    skipped_interactions += 1
                    continue

                for p_name in participants: agents.add(p_name)

                # --- 计算交互并收集对话文本 ---
                if len(participants) >= 2:
                    for i in range(len(participants)):
                        for j in range(i + 1, len(participants)):
                            agent1, agent2 = sorted([participants[i], participants[j]])
                            pair_key = (agent1, agent2)

                            # 1. 增加交互计数
                            interaction_counts[pair_key] += 1

                            # 2. 收集该次对话的文本
                            # 将 [[speaker, text], ...] 格式化为更易读的字符串列表
                            formatted_chat = [f"{speaker}: {text}" for speaker, text in chat_log]
                            pairwise_conversations[pair_key].extend(formatted_chat)
                # else: 处理单人情况 (如果需要)

            except Exception as e:
                print(f"错误：处理{timestamp}的键'{persons_location_key}'时发生意外错误：{e}")
                skipped_interactions += 1
                continue

    # 打印处理总结 (与之前类似)
    print(f"处理完成。")
    print(f"  处理的交互条目总数：{processed_interactions}")
    print(f"  由于解析问题跳过的交互：{skipped_interactions}")
    print(f"  找到{len(agents)}个唯一代理：{', '.join(sorted(list(agents)))}")
    print(f"  计数{len(interaction_counts)}个唯一的成对交互。")
    print(f"  收集了{len(pairwise_conversations)}对代理的对话记录。")

    return agents, interaction_counts, pairwise_conversations


# --- 使用 LLM 生成对话总结 ---
def generate_summaries(pairwise_conversations, llm_instance: LLMModel):
    """使用提供的LLM实例为每对agent的对话生成总结。"""
    if not llm_instance or not llm_instance.is_available():
        print("LLM 实例不可用，跳过总结生成。")
        return {}

    edge_summaries = {}
    total_pairs = len(pairwise_conversations)
    print(f"开始为 {total_pairs} 对交互生成对话总结...")

    for i, (pair, conversations) in enumerate(pairwise_conversations.items()):
        agent1, agent2 = pair
        print(f"  正在处理第 {i + 1}/{total_pairs} 对: ({agent1}, {agent2})... ", end="")

        # 将对话列表合并成一个大字符串
        combined_chat_log = "\n".join(conversations)

        # --- 构建 Prompt ---
        # 可以根据需要调整这个 prompt
        prompt = f"""请仔细阅读以下 '{agent1}' 和 '{agent2}' 之间的对话记录。

对话记录：
--- START ---
{combined_chat_log}
--- END ---

请根据上述对话，用一两句话简洁地总结他们讨论的核心主题、主要观点或达成的共识。总结应清晰明了，避免无关细节，适合用作关系图中交互的标签。
如果对话内容简单或无实质信息，请指出（例如：“简单问候”或“未涉及具体话题”）。

总结："""

        # 调用 LLM
        try:
            # 使用 llm_model.py 中的 completion 方法
            # 设置较低的 retry 次数以避免长时间等待，或根据需要调整
            summary = llm_instance.completion(
                prompt=prompt,
                retry=3,  # 减少重试次数
                failsafe="无法生成总结。",  # 失败时的备用文本
                caller="summarization",  # 用于统计
                temperature=0.2  # 较低的温度可能使总结更集中
            )

            if summary and summary != "无法生成总结。":
                edge_summaries[pair] = summary.strip()
                print("完成。")
            else:
                edge_summaries[pair] = "未能获取有效总结。"  # 明确标记失败
                print("失败或总结为空。")

            # 可选：在每次 LLM 调用后稍微暂停，以避免触发速率限制
            # time.sleep(1)

        except Exception as e:
            print(f"LLM 调用失败：{e}")
            edge_summaries[pair] = f"总结生成出错: {e}"

    print(f"对话总结生成完成。成功为 {len(edge_summaries)} 对交互生成了总结。")
    # print("LLM 调用统计:", llm_instance.get_summary()) # 打印 LLM 调用统计
    return edge_summaries


# --- 图的创建和可视化 (集成总结) ---
def create_and_visualize_network_with_summary(agents, interaction_counts, edge_summaries, output_filename):
    """使用PyVis创建NetworkX图并将其可视化，包含对话总结。"""
    if not agents:
        print("未找到代理来创建网络。")
        return

    G = nx.Graph()
    agent_list = sorted(list(agents))
    for agent in agent_list:
        G.add_node(agent, label=agent, title=agent)

    total_interaction_weight = 0
    edges_added = 0
    for (agent1, agent2), weight in interaction_counts.items():
        if agent1 in agents and agent2 in agents:
            pair_key = (agent1, agent2)  # key 是排序后的
            # 获取总结，如果不存在则使用默认提示
            summary = edge_summaries.get(pair_key, "无对话总结。")

            # 构建增强的 title (悬停提示)
            edge_title = f"<b>交互次数:</b> {weight}<br><br>" \
                         f"<b>对话概要:</b><br>{summary.replace(chr(10), '<br>')}"  # 替换换行符为HTML换行

            G.add_edge(agent1, agent2, weight=weight, title=edge_title)
            total_interaction_weight += weight
            edges_added += 1
        # else: 跳过无效边 (与之前相同)

    if G.number_of_edges() == 0 and edges_added == 0:
        print("没有将交互边添加到图中。无法可视化网络图。")
        return

    print(f"创建了具有{G.number_of_nodes()}个节点和{G.number_of_edges()}条边的图。")
    print(f"总交互计数（边权重之和）：{total_interaction_weight}")

    # --- 创建 PyVis 网络 (大部分与之前相同) ---
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, filter_menu=True,
                  select_menu=True)
    net.from_nx(G)  # 从 NetworkX 加载，继承节点和边的属性 (包括 title)

    net.barnes_hut(gravity=-5000, central_gravity=0.1, spring_length=150, spring_strength=0.01, damping=0.09,
                   overlap=0.1)
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    # --- 设置边和节点的视觉属性 (使用继承的属性) ---
    # PyVis 会自动使用 'value' (如果存在) 来缩放边宽，使用 'size' 缩放节点
    # 我们需要在 net.from_nx 之后，根据 G 中的 weight 和 degree 设置 PyVis 对象的属性

    edge_weights = {}  # 重新获取权重，因为 net.from_nx 可能不直接映射 weight 到 value
    for u, v, data in G.edges(data=True):
        key = f"{u};{v}"
        rev_key = f"{v};{u}"
        edge_weights[key] = data.get('weight', 1)
        edge_weights[rev_key] = data.get('weight', 1)

    for edge in net.edges:
        source = edge.get('from')
        target = edge.get('to')
        if source and target:
            key = f"{source};{target}"
            weight = edge_weights.get(key, 1)
            edge['value'] = weight  # 设置 value 用于厚度缩放
            # title 应该已经从 G 中继承过来了

    degrees = dict(G.degree(weight='weight'))
    if degrees:
        max_degree = max(degrees.values()) if degrees else 1.0
        min_degree = min(degrees.values()) if degrees else 1.0
        for node in net.nodes:
            node_id = node['id']
            node_degree = degrees.get(node_id, 0)
            node['size'] = 10 + 20 * (node_degree - min_degree) / (max_degree - min_degree + 1e-6)
            # title 应该已经从 G 继承

    # --- 生成并保存 HTML (与之前相同) ---
    try:
        net.save_graph(output_filename)
        print(f"成功生成网络图：'{output_filename}'")
        try:
            webbrowser.open('file://' + os.path.realpath(output_filename))
        except Exception as e:
            print(f"无法在浏览器中自动打开文件：{e}")
            print(f"请在您的网络浏览器中手动打开'{output_filename}'。")
    except Exception as e:
        print(f"保存或生成图HTML时出错：{e}")


# --- 主执行 ---
if __name__ == "__main__":
    # 1. 加载对话数据
    conversation_data = load_conversation_data(CONVERSATION_FILE_PATH)

    if conversation_data:
        # 2. 处理交互并收集对话
        agents, interaction_counts, pairwise_conversations = process_interactions_and_conversations(conversation_data)

        edge_summaries = {}
        llm_instance = None

        # 3. 如果 LLM 可用且启用，则尝试创建实例并生成总结
        if LLM_AVAILABLE and LLM_CONFIG.get("enable_summarization", True):
            print("\n--- 开始 LLM 初始化和总结生成 ---")
            try:
                # 使用配置创建 LLM 实例
                print("正在创建 LLM 实例...")
                llm_instance = create_llm_model(
                    base_url=LLM_CONFIG["base_url"],
                    model=LLM_CONFIG["model"],
                    embedding_model=LLM_CONFIG.get("embedding_model"),
                    keys=LLM_CONFIG["keys"]
                )
                print(f"LLM 实例 ({llm_instance.__class__.__name__}) 创建成功。")

                # 生成总结
                if llm_instance.is_available():
                    edge_summaries = generate_summaries(pairwise_conversations, llm_instance)
                else:
                    print("LLM 实例已创建但不可用（可能初始化失败或被禁用）。跳过总结。")

            except Exception as e:
                print(f"初始化或运行 LLM 时发生错误：{e}")
            print("--- LLM 处理结束 ---\n")
        else:
            print("\nLLM 总结功能已禁用或不可用。\n")

        # 4. 创建并可视化网络
        if agents:
            create_and_visualize_network_with_summary(
                agents,
                interaction_counts,
                edge_summaries,
                OUTPUT_HTML_FILE
            )
        else:
            print("处理完成，但未识别出代理。无法创建网络。")
