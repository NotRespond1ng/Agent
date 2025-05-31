"""generative_agents.memory.associate"""

import datetime
import os
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

from modules.storage.index import LlamaIndex
from modules import utils
from .event import Event


class Concept:
    def __init__(
        self,
        describe,
        node_id,
        node_type,
        subject,
        predicate,
        object,
        address,
        poignancy,
        create=None,
        expire=None,
        access=None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.event = Event(
            subject, predicate, object, describe=describe, address=address.split(":")
        )
        self.poignancy = poignancy
        self.create = utils.to_date(create) if create else utils.get_timer().get_date()
        if expire:
            self.expire = utils.to_date(expire)
        else:
            self.expire = self.create + datetime.timedelta(days=30)
        self.access = utils.to_date(access) if access else self.create

    def abstract(self):
        return {
            "{}(P.{})".format(self.node_type, self.poignancy): str(self.event),
            "duration": "{} ~ {} (access: {})".format(
                self.create.strftime("%Y%m%d-%H:%M"),
                self.expire.strftime("%Y%m%d-%H:%M"),
                self.access.strftime("%Y%m%d-%H:%M"),
            ),
        }

    def __str__(self):
        return utils.dump_dict(self.abstract())

    @property
    def describe(self):
        return self.event.get_describe()

    @classmethod
    def from_node(cls, node):
        return cls(node.text, node.id_, **node.metadata)

    @classmethod
    def from_event(cls, node_id, node_type, event, poignancy):
        return cls(
            event.get_describe(),
            node_id,
            node_type,
            event.subject,
            event.predicate,
            event.object,
            ":".join(event.address),
            poignancy,
        )


class AssociateRetriever(BaseRetriever):
    def __init__(self, config, *args, **kwargs) -> None:
        self._config = config
        self._vector_retriever = VectorIndexRetriever(*args, **kwargs)
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve nodes given query."""

        nodes = self._vector_retriever.retrieve(query_bundle)
        if not nodes:
            return []
        nodes = sorted(
            nodes, key=lambda n: utils.to_date(n.metadata["access"]), reverse=True
        )
        # get scores
        fac = self._config["recency_decay"]
        recency_scores = self._normalize(
            [fac**i for i in range(1, len(nodes) + 1)], self._config["recency_weight"]
        )
        relevance_scores = self._normalize(
            [n.score for n in nodes], self._config["relevance_weight"]
        )
        importance_scores = self._normalize(
            [n.metadata["poignancy"] for n in nodes], self._config["importance_weight"]
        )
        final_scores = {
            n.id_: r1 + r2 + i
            for n, r1, r2, i in zip(
                nodes, recency_scores, relevance_scores, importance_scores
            )
        }
        # re-rank nodes
        nodes = sorted(nodes, key=lambda n: final_scores[n.id_], reverse=True)
        nodes = nodes[: self._config["retrieve_max"]]
        for n in nodes:
            n.metadata["access"] = utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        return nodes

    def _normalize(self, data, factor=1, t_min=0, t_max=1):
        min_val, max_val = min(data), max(data)
        diff = max_val - min_val
        if diff == 0:
            return [(t_max - t_min) * factor / 2 for _ in data]
        return [(d - min_val) * (t_max - t_min) * factor / diff + t_min for d in data]


class Associate:
    def __init__(
        self,
        path,
        embedding,
        retention=8,
        max_memory=-1,
        max_importance=10,
        recency_decay=0.995,
        recency_weight=0.5,
        relevance_weight=3,
        importance_weight=2,
        memory=None,
    ):
        self._index_config = {"embedding": embedding, "path": path}
        print(f"正在初始化Associate，路径: {path}")
        
        # 初始化LlamaIndex
        self._index = LlamaIndex(**self._index_config)
        print(f"LlamaIndex初始化完成，当前节点数: {self._index.nodes_num}")
        
        # 尝试从存储中恢复memory数据
        if memory is None and path and os.path.exists(path):
            memory_file = os.path.join(path, "memory.json")
            if os.path.exists(memory_file):
                try:
                    import json
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        memory = saved_data.get("memory", {"event": [], "thought": [], "chat": []})
                        print(f"从 {memory_file} 恢复记忆数据: {len(memory.get('chat', []))} 对话, {len(memory.get('event', []))} 事件")
                except Exception as e:
                    print(f"加载记忆数据失败: {e}")
                    memory = {"event": [], "thought": [], "chat": []}
        
        self.memory = memory or {"event": [], "thought": [], "chat": []}
        
        # 在索引完全加载后再进行清理
        print(f"索引加载完成后的节点数: {self._index.nodes_num}")
        if self._index.nodes_num > 0:
            print("执行索引清理...")
            self.cleanup_index()
            print(f"清理后的节点数: {self._index.nodes_num}")
        else:
            print("索引为空，跳过清理步骤")
        self.retention = retention
        self.max_memory = max_memory
        self.max_importance = max_importance
        self._retrieve_config = {
            "recency_decay": recency_decay,
            "recency_weight": recency_weight,
            "relevance_weight": relevance_weight,
            "importance_weight": importance_weight,
        }
        # 添加自动保存计数器
        self._save_counter = 0
        self._save_interval = 5  # 每添加5个节点自动保存一次

    def abstract(self):
        des = {"nodes": self._index.nodes_num}
        for t in ["event", "chat", "thought"]:
            des[t] = [self.find_concept(c).describe for c in self.memory[t]]
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def cleanup_index(self):
        node_ids = self._index.cleanup()
        self.memory = {
            n_type: [n for n in nodes if n not in node_ids]
            for n_type, nodes in self.memory.items()
        }

    def add_node(
        self,
        node_type,
        event,
        poignancy,
        create=None,
        expire=None,
        filling=None,
    ):
        create = create or utils.get_timer().get_date()
        expire = expire or (create + datetime.timedelta(days=30))
        metadata = {
            "node_type": node_type,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "address": ":" .join(event.address),
            "poignancy": poignancy,
            "create": create.strftime("%Y%m%d-%H:%M:%S"),
            "expire": expire.strftime("%Y%m%d-%H:%M:%S"),
            "access": create.strftime("%Y%m%d-%H:%M:%S"),
        }
        node = self._index.add_node(event.get_describe(), metadata)
        memory = self.memory[node_type]
        memory.insert(0, node.id_)
        if len(memory) >= self.max_memory > 0:
            self._index.remove_nodes(memory[self.max_memory:])
            self.memory[node_type] = memory[: self.max_memory - 1]
        
        # 添加自动保存逻辑
        self._save_counter += 1
        if self._save_counter >= self._save_interval:
            self._index.save()
            self._save_counter = 0
        
        return self.to_concept(node)

    def to_concept(self, node):
        return Concept.from_node(node)

    def find_concept(self, node_id):
        return self.to_concept(self._index.find_node(node_id))

    def _retrieve_nodes(self, node_type, text=None):
        print(f"\n=== _retrieve_nodes 调试信息 ===")
        print(f"node_type: {node_type}")
        print(f"text: {text}")
        print(f"self.memory[{node_type}]: {self.memory[node_type]}")
        print(f"self.memory[{node_type}] 长度: {len(self.memory[node_type])}")
        
        if text:
            print(f"\n使用文本检索模式: {text}")
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="node_type", value=node_type)]
            )
            print(f"过滤器: {filters}")
            
            # 检查指定的node_ids是否在索引中存在
            print(f"\n检查node_ids在索引中的存在性:")
            existing_nodes = []
            for node_id in self.memory[node_type]:
                if self._index.has_node(node_id):
                    existing_nodes.append(node_id)
                    node = self._index.find_node(node_id)
                    print(f"  ✓ {node_id}: 存在, 文本长度={len(node.text) if hasattr(node, 'text') else 'N/A'}")
                    # 检查节点的嵌入状态
                    if hasattr(node, 'embedding') and node.embedding is not None:
                        print(f"    嵌入: 已存在 (维度: {len(node.embedding)})")
                    else:
                        print(f"    嵌入: 未生成或为空")
                else:
                    print(f"  ✗ {node_id}: 不存在于索引中")
            
            print(f"\n有效的node_ids数量: {len(existing_nodes)}/{len(self.memory[node_type])}")
            
            # 调用LlamaIndex的retrieve方法
            print(f"\n调用 self._index.retrieve...")
            print(f"  参数: text='{text}', filters={filters}, node_ids={self.memory[node_type]}")
            
            try:
                nodes = self._index.retrieve(
                    text, filters=filters, node_ids=self.memory[node_type]
                )
                print(f"  ✓ retrieve成功返回 {len(nodes)} 个节点")
                
                if len(nodes) == 0:
                    print(f"  ⚠️  警告: retrieve返回空结果")
                    print(f"  可能原因:")
                    print(f"    1. 查询文本'{text}'与节点内容语义不匹配")
                    print(f"    2. 嵌入模型未正确生成或加载节点嵌入")
                    print(f"    3. node_ids参数处理有问题")
                    print(f"    4. 过滤器配置错误")
                    
                    # 尝试不使用node_ids参数进行检索
                    print(f"\n  尝试不使用node_ids参数的检索:")
                    try:
                        nodes_without_ids = self._index.retrieve(text, filters=filters)
                        print(f"    无node_ids检索结果: {len(nodes_without_ids)} 个节点")
                        if len(nodes_without_ids) > 0:
                            print(f"    说明问题可能在node_ids参数处理")
                        else:
                            print(f"    说明问题可能在嵌入生成或查询匹配")
                    except Exception as e:
                        print(f"    无node_ids检索也失败: {e}")
                        
                    # 尝试不使用过滤器进行检索
                    print(f"\n  尝试不使用过滤器的检索:")
                    try:
                        nodes_without_filters = self._index.retrieve(text, node_ids=self.memory[node_type])
                        print(f"    无过滤器检索结果: {len(nodes_without_filters)} 个节点")
                        if len(nodes_without_filters) > 0:
                            print(f"    说明问题可能在过滤器配置")
                    except Exception as e:
                        print(f"    无过滤器检索也失败: {e}")
                        
                else:
                    for i, node in enumerate(nodes[:3]):  # 显示前3个结果
                        print(f"    节点{i+1}: id={node.id_}, 文本长度={len(node.text) if hasattr(node, 'text') else 'N/A'}")
                        if hasattr(node, 'score'):
                            print(f"      相似度分数: {node.score}")
                            
            except Exception as e:
                print(f"  ✗ retrieve调用失败: {e}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
                nodes = []
        else:
            print(f"\n使用直接查找模式 (无文本查询)")
            nodes = []
            for node_id in self.memory[node_type]:
                try:
                    node = self._index.find_node(node_id)
                    nodes.append(node)
                    print(f"  ✓ 找到节点: {node_id}")
                except Exception as e:
                    print(f"  ✗ 找不到节点: {node_id}, 错误: {e}")
            print(f"直接查找模式返回 {len(nodes)} 个节点")
            
        print(f"\n最终返回节点数: {len(nodes)}")
        result = [self.to_concept(n) for n in nodes[: self.retention]]
        print(f"转换为概念后数量: {len(result)}")
        print(f"=== _retrieve_nodes 调试完成 ===\n")
        return result

    def retrieve_events(self, text=None):
        return self._retrieve_nodes("event", text)

    def retrieve_thoughts(self, text=None):
        return self._retrieve_nodes("thought", text)

    def retrieve_chats(self, name=None):
        text = ("对话 " + name) if name else None
        return self._retrieve_nodes("chat", text)

    def retrieve_focus(self, focus, retrieve_max=30, reduce_all=True):
        def _create_retriever(*args, **kwargs):
            self._retrieve_config["retrieve_max"] = retrieve_max
            return AssociateRetriever(self._retrieve_config, *args, **kwargs)

        retrieved = {}
        node_ids = self.memory["event"] + self.memory["thought"]
        for text in focus:
            nodes = self._index.retrieve(
                text,
                similarity_top_k=len(node_ids),
                node_ids=node_ids,
                retriever_creator=_create_retriever,
            )
            if reduce_all:
                retrieved.update({n.id_: n for n in nodes})
            else:
                retrieved[text] = nodes
        if reduce_all:
            return [self.to_concept(v) for v in retrieved.values()]
        return {
            text: [self.to_concept(n) for n in nodes]
            for text, nodes, in retrieved.items()
        }

    def get_relation(self, node):
        return {
            "node": node,
            "events": self.retrieve_events(node.describe),
            "thoughts": self.retrieve_thoughts(node.describe),
        }

    def to_dict(self):
        # 确保在序列化时保存索引
        self._index.save()
        
        # 同时将memory数据保存到单独的文件中
        if self._index_config.get("path"):
            memory_file = os.path.join(self._index_config["path"], "memory.json")
            try:
                import json
                os.makedirs(os.path.dirname(memory_file), exist_ok=True)
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump({"memory": self.memory}, f, ensure_ascii=False, indent=2)
                print(f"保存记忆数据到 {memory_file}: {len(self.memory.get('chat', []))} 对话, {len(self.memory.get('event', []))} 事件")
            except Exception as e:
                print(f"保存记忆数据失败: {e}")
        
        return {"memory": self.memory}

    @property
    def index(self):
        return self._index
