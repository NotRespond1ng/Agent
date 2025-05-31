"""generative_agents.storage.index"""

import os
import time
import datetime
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index import core as index_core
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
# from langchain_openai import OpenAIEmbeddings

from modules import utils


class LlamaIndex:
    def __init__(self, embedding, path=None):
        self._config = {"max_nodes": 0}
        if embedding["type"] == "hugging_face":
            embed_model = HuggingFaceEmbedding(model_name=embedding["model"])
        elif embedding["type"] == "ollama":
            embed_model = OllamaEmbedding(
                model_name=embedding["model"],
                base_url=embedding["base_url"],
                ollama_additional_kwargs={"mirostat": 0},
            )
        elif embedding['type'] == 'openai':
            embed_model = OpenAIEmbedding(
                model=embedding['model'],
                api_base=embedding['base_url'],
                api_key='sk-Vlr3Ie1XffJP1hFyPYH9op021Oa6y0tbRgU6HOIdpkUkFTJm'
            )
        else:
            raise NotImplementedError(
                "embedding type {} is not supported".format(embedding["type"])
            )

        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        Settings.num_output = 1024
        Settings.context_window = 4096
        if path and os.path.exists(path):
            print(f"\n=== LlamaIndex 初始化开始 ===")
            print(f"正在从路径加载LlamaIndex: {path}")
            
            # 检查关键文件是否存在
            docstore_path = os.path.join(path, "docstore.json")
            index_path = os.path.join(path, "index_store.json")
            vector_path = os.path.join(path, "vector_store.json")
            
            print(f"检查存储文件:")
            print(f"  docstore.json: {'存在' if os.path.exists(docstore_path) else '不存在'}")
            print(f"  index_store.json: {'存在' if os.path.exists(index_path) else '不存在'}")
            print(f"  vector_store.json: {'存在' if os.path.exists(vector_path) else '不存在'}")
            
            if os.path.exists(docstore_path):
                print(f"\n读取docstore.json文件: {docstore_path}")
                # 读取并检查docstore.json内容
                try:
                    import json
                    with open(docstore_path, 'r', encoding='utf-8') as f:
                        docstore_data = json.load(f)
                        print(f"  docstore.json结构: {list(docstore_data.keys())}")
                        if 'docstore/data' in docstore_data:
                            node_count = len(docstore_data['docstore/data'])
                            print(f"  ✓ docstore.json包含 {node_count} 个节点")
                            
                            # 显示前几个节点的ID
                            node_ids = list(docstore_data['docstore/data'].keys())[:3]
                            print(f"  前3个节点ID: {node_ids}")
                        else:
                            print(f"  ✗ docstore.json格式异常，未找到docstore/data字段")
                            print(f"  实际字段: {list(docstore_data.keys())}")
                except Exception as e:
                    print(f"  ✗ 读取docstore.json失败: {e}")
            else:
                print(f"\n✗ 警告: docstore.json文件不存在于 {path}")
            
            print(f"\n开始加载LlamaIndex存储...")
            try:
                storage_context = index_core.StorageContext.from_defaults(persist_dir=path)
                print(f"  ✓ 成功创建StorageContext")
                print(f"  StorageContext.docstore类型: {type(storage_context.docstore)}")
                
                # 检查StorageContext中的docstore
                if hasattr(storage_context.docstore, 'docs'):
                    print(f"  StorageContext.docstore包含 {len(storage_context.docstore.docs)} 个文档")
                
                self._index = index_core.load_index_from_storage(
                    storage_context,
                    show_progress=True,
                )
                print(f"  ✓ 成功加载索引")
                print(f"  索引类型: {type(self._index)}")
                
                # 详细检查加载后的索引状态
                if hasattr(self._index, 'docstore'):
                    docstore_docs_count = len(self._index.docstore.docs)
                    print(f"  ✓ 索引docstore包含 {docstore_docs_count} 个文档")
                    
                    if docstore_docs_count > 0:
                        # 显示前几个文档的信息
                        doc_ids = list(self._index.docstore.docs.keys())[:3]
                        print(f"  前3个文档ID: {doc_ids}")
                        
                        for doc_id in doc_ids[:1]:  # 只显示第一个文档的详细信息
                            doc = self._index.docstore.docs[doc_id]
                            print(f"    文档 {doc_id}: 类型={type(doc)}, 文本长度={len(doc.text) if hasattr(doc, 'text') else 'N/A'}")
                    else:
                        print(f"  ⚠️  警告: 索引加载成功但docstore为空")
                else:
                    print(f"  ✗ 索引没有docstore属性")
                    
            except Exception as e:
                print(f"  ✗ 加载索引失败: {e}")
                print(f"  错误类型: {type(e)}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
                print(f"  创建新的空索引")
                self._index = index_core.VectorStoreIndex([], show_progress=True)
            
            config_path = os.path.join(path, "index_config.json")
            if os.path.exists(config_path):
                self._config = utils.load_dict(config_path)
                print(f"  ✓ 加载配置文件: {config_path}")
            else:
                print(f"  ⚠️  警告: index_config.json文件不存在于 {path}")
                
            print(f"=== LlamaIndex 初始化完成 ===\n")
        else:
            if path:
                print(f"✗ 指定路径不存在: {path}")
            else:
                print(f"未指定路径，创建空索引")
            self._index = index_core.VectorStoreIndex([], show_progress=True)
        self._path = path

    def add_node(
        self,
        text,
        metadata=None,
        exclude_llm_keys=None,
        exclude_embedding_keys=None,
        id=None,
    ):
        while True:
            try:
                metadata = metadata or {}
                exclude_llm_keys = exclude_llm_keys or list(metadata.keys())
                exclude_embedding_keys = exclude_embedding_keys or list(metadata.keys())
                id = id or "node_" + str(self._config["max_nodes"])
                self._config["max_nodes"] += 1
                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    excluded_llm_metadata_keys=exclude_llm_keys,
                    excluded_embed_metadata_keys=exclude_embedding_keys,
                )
                self._index.insert_nodes([node])
                return node
            except Exception as e:
                print(f"LlamaIndex.add_node() caused an error: {e}")
                time.sleep(5)

    def has_node(self, node_id):
        return node_id in self._index.docstore.docs

    def find_node(self, node_id):
        return self._index.docstore.docs[node_id]

    def get_nodes(self, filter=None):
        def _check(node):
            if not filter:
                return True
            return filter(node)

        return [n for n in self._index.docstore.docs.values() if _check(n)]

    def remove_nodes(self, node_ids, delete_from_docstore=True):
        self._index.delete_nodes(node_ids, delete_from_docstore=delete_from_docstore)

    def cleanup(self):
        print(f"\n=== 开始索引清理 ===")
        now = utils.get_timer().get_date()
        remove_ids = []
        print(f"当前时间: {now}")
        
        total_nodes = len(self._index.docstore.docs)
        print(f"清理前节点总数: {total_nodes}")
        
        if total_nodes == 0:
            print(f"没有节点需要清理")
            print(f"=== 索引清理完成 ===\n")
            return remove_ids
        
        # 检查前几个节点的时间信息
        sample_nodes = list(self._index.docstore.docs.items())[:3]
        for node_id, node in sample_nodes:
            print(f"\n节点 {node_id} 的metadata:")
            if hasattr(node, 'metadata') and node.metadata:
                print(f"  metadata keys: {list(node.metadata.keys())}")
                if 'create' in node.metadata:
                    print(f"  create: {node.metadata['create']} (类型: {type(node.metadata['create'])})")
                if 'expire' in node.metadata:
                    print(f"  expire: {node.metadata['expire']} (类型: {type(node.metadata['expire'])})")
            else:
                print(f"  节点没有metadata或metadata为空")
        
        for node_id, node in self._index.docstore.docs.items():
            try:
                # 检查节点是否有必要的时间字段
                if not hasattr(node, 'metadata') or not node.metadata:
                    print(f"警告: 节点 {node_id} 没有metadata，跳过清理")
                    continue
                    
                if 'create' not in node.metadata or 'expire' not in node.metadata:
                    print(f"警告: 节点 {node_id} 缺少时间字段，跳过清理")
                    continue
                
                # 尝试解析时间
                create_str = node.metadata["create"]
                expire_str = node.metadata["expire"]
                
                # 如果时间字段已经是datetime对象，直接使用
                if isinstance(create_str, datetime.datetime):
                    create = create_str
                else:
                    create = utils.to_date(create_str)
                    
                if isinstance(expire_str, datetime.datetime):
                    expire = expire_str
                else:
                    expire = utils.to_date(expire_str)
                
                # 检查是否需要移除
                # 注意：暂时禁用基于过期时间的清理，因为历史数据的过期时间可能早于当前时间
                # 只有当创建时间晚于当前时间时才删除（这种情况通常不应该发生）
                if create > now:
                    print(f"节点 {node_id} 创建时间 {create} 晚于当前时间 {now}，标记删除")
                    remove_ids.append(node_id)
                # elif expire < now:
                #     print(f"节点 {node_id} 过期时间 {expire} 早于当前时间 {now}，标记删除")
                #     remove_ids.append(node_id)
                else:
                    # 保留节点，不进行基于过期时间的清理
                    pass
                    
            except Exception as e:
                print(f"处理节点 {node_id} 时出错: {e}，跳过清理")
                continue
        
        print(f"\n标记删除的节点数: {len(remove_ids)}")
        if remove_ids:
            print(f"删除的节点ID: {remove_ids[:5]}{'...' if len(remove_ids) > 5 else ''}")
            self.remove_nodes(remove_ids)
        
        remaining_nodes = len(self._index.docstore.docs)
        print(f"清理后节点数: {remaining_nodes}")
        print(f"=== 索引清理完成 ===\n")
        return remove_ids

    def retrieve(
        self,
        text,
        similarity_top_k=5,
        filters=None,
        node_ids=None,
        retriever_creator=None,
    ):
        while True:
            try:
                retriever_creator = retriever_creator or VectorIndexRetriever
                return retriever_creator(
                    self._index,
                    similarity_top_k=similarity_top_k,
                    filters=filters,
                    node_ids=node_ids,
                ).retrieve(text)
            except Exception as e:
                print(f"LlamaIndex.retrieve() caused an error: {e}")
                time.sleep(5)

    def query(
        self,
        text,
        similarity_top_k=5,
        text_qa_template=None,
        refine_template=None,
        filters=None,
        query_creator=None,
    ):
        kwargs = {
            "similarity_top_k": similarity_top_k,
            "text_qa_template": text_qa_template,
            "refine_template": refine_template,
            "filters": filters,
        }
        while True:
            try:
                if query_creator:
                    query_engine = query_creator(retriever=self._index.as_retriever(**kwargs))
                else:
                    query_engine = self._index.as_query_engine(**kwargs)
                return query_engine.query(text)
            except Exception as e:
                print(f"LlamaIndex.query() caused an error: {e}")
                time.sleep(5)

    def save(self, path=None):
        path = path or self._path
        self._index.storage_context.persist(path)
        utils.save_dict(self._config, os.path.join(path, "index_config.json"))

    @property
    def nodes_num(self):
        return len(self._index.docstore.docs)
