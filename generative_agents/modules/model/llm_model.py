"""generative_agents.model.llm_model"""

import os
import time
import re
import json
import requests

# 导入自定义模块
from modules.utils.namespace import ModelType
from modules import utils


class ModelStyle:
    """模型风格定义"""

    OPEN_AI = "openai"
    QIANFAN = "qianfan"
    SPARK_AI = "sparkai"
    ZHIPU_AI = "zhipuai"
    OLLAMA = "ollama"


class LLMModel:
    """大语言模型基类"""
    def __init__(self, base_url, model, embedding_model, keys, config=None):
        # 初始化函数
        self._base_url = base_url  # API 的基础 URL (例如 OpenAI 的 https://api.openai.com/v1 或代理地址)
        self._model = model  # 用于生成文本补全的具体模型名称 (例如 "gpt-4o-mini")
        # 用于文本嵌入的具体模型名称 (例如 "text-embedding-3-small")，由参数传入并在此设置
        self._embedding_model = embedding_model
        # 调用 setup 方法初始化 API 客户端或句柄 (handle)
        # keys 包含 API 密钥, config 是可选的额外配置 (在此场景下通常为 None)
        self._handle = self.setup(keys, config)
        self._meta_responses = []  # 存储原始 API 响应的列表 (用于调试或记录)
        self._summary = {"total": [0, 0, 0]}  # 记录调用统计 [总尝试次数, 成功次数, 失败次数]
        self._enabled = True  # 模型是否启用

    def embedding(self, text, retry=10):
        """生成文本嵌入向量 (带重试逻辑)"""
        response = None
        for _ in range(retry):  # 尝试多次
            try:
                # 调用具体实现的 _embedding 方法
                response = self._embedding(text)
            except Exception as e:
                # 捕获异常，打印错误信息，等待后重试
                print(f"LLMModel.embedding() 发生错误: {e}")
                time.sleep(5)
                continue
            if response:  # 如果成功获取响应，则跳出循环
                break
        return response  # 返回嵌入向量或 None

    def _embedding(self, text):
        """生成文本嵌入向量 (具体实现)"""
        # 如果子类支持 embedding 且已正确初始化 handle (例如 OpenAI 客户端)
        if self._handle and hasattr(self._handle, 'embeddings'):
             try:
                 # 使用初始化好的 handle 和 _embedding_model 调用嵌入 API
                 response = self._handle.embeddings.create(
                     input=text, model=self._embedding_model
                 )
                 # 检查响应数据是否有效
                 if response.data and len(response.data) > 0:
                     return response.data[0].embedding  # 返回第一个嵌入向量
                 else:
                     # 记录警告信息
                     print(f"警告: Embedding 响应没有数据，输入文本: {text[:50]}...")
                     return None
             except Exception as e:
                 # 捕获并打印嵌入生成过程中的错误
                 print(f"错误: Embedding 生成过程中出错: {e}")
                 # 重新抛出异常，让外部的 embedding 方法的重试逻辑处理
                 raise e
        else:
            # 如果 handle 无效或子类未实现 _embedding，则抛出未实现错误
            raise NotImplementedError(
                f"_embedding 需要一个有效的、支持 embedding 的句柄 (handle)，或者在子类 {self.__class__} 中被重写"
            )

    def completion(
        self,
        prompt,
        retry=10,
        callback=None,
        failsafe=None,
        caller="llm_normal",
        **kwargs
    ):
        """生成文本补全 (带重试逻辑)"""
        response, self._meta_responses = None, [] # 初始化响应和元响应列表
        self._summary.setdefault(caller, [0, 0, 0]) # 初始化调用者统计

        # 检查句柄是否已初始化
        if not self._handle:
             print(f"错误: {self.__class__.__name__} 的 LLM 句柄 (handle) 未初始化。跳过 completion 调用。")
             return failsafe # 如果句柄无效，返回备用值

        for i in range(retry): # 尝试多次
            try:
                # 调用具体实现的 _completion 方法，传递 prompt 和其他参数
                meta_response_content = self._completion(prompt, **kwargs)

                # 存储原始响应内容 (假设 _completion 返回字符串内容)
                self._meta_responses.append(str(meta_response_content) if meta_response_content is not None else "<Completion Failed>")

                # 检查补全是否成功 (返回非 None 或非空字符串)
                if meta_response_content is not None and meta_response_content != "":
                    self._summary["total"][0] += 1 # 总尝试次数 +1
                    self._summary[caller][0] += 1 # 调用者尝试次数 +1
                    if callback:
                        # 如果有回调函数，则调用它
                        response = callback(meta_response_content)
                    else:
                        # 否则直接使用返回的内容
                        response = meta_response_content
                    break # 成功则跳出重试循环
                else:
                     # 处理 _completion 返回 None 或空字符串的情况 (没有抛出异常)
                     print(f"LLMModel.completion() 尝试 {i+1}/{retry} 返回为空或 None。")
                     response = None # 确保 response 为 None 以便重试逻辑判断
                     # 可选：即使没有异常，也稍微延迟一下再重试
                     # time.sleep(1)

            except Exception as e:
                # 捕获 _completion 调用或后续处理中发生的异常
                print(f"LLMModel.completion() 尝试 {i+1}/{retry} 发生错误: {e}")
                # 在元响应中记录错误信息
                self._meta_responses.append(f"<Error: {e}>")
                time.sleep(5) # 等待 5 秒
                response = None # 标记为失败
                # 继续下一次重试迭代
                continue

        # 根据最终结果更新统计 (如果 response 为 None 则视为失败)
        pos = 2 if response is None else 1 # 1: 成功索引, 2: 失败索引
        self._summary["total"][pos] += 1 # 总成功/失败次数 +1
        self._summary[caller][pos] += 1 # 调用者成功/失败次数 +1
        # 返回最终响应，如果失败则返回备用值 (failsafe)
        return response or failsafe

    def _completion(self, prompt, **kwargs):
        """生成文本补全 (具体实现) - 应由子类重写"""
        raise NotImplementedError(
            f"_completion 在子类 {self.__class__} 中未实现"
        )

    def is_available(self):
        """检查模型是否可用"""
        # 模型已启用且句柄 (handle) 已成功初始化
        return self._enabled and self._handle is not None

    def get_summary(self):
        """获取调用统计摘要"""
        des = {}
        for k, v in self._summary.items():
            # 格式: S:成功次数, F:失败次数 / R:总尝试次数
            des[k] = "S:{},F:{}/R:{}".format(v[1], v[2], v[0])
        return {"model": self._model, "summary": des}

    def disable(self):
        """禁用模型"""
        self._enabled = False

    @property
    def meta_responses(self):
        """获取存储的原始 API 响应列表"""
        return self._meta_responses

    @classmethod
    def model_type(cls):
        """返回模型类型"""
        return ModelType.LLM


# --- OpenAI 模型实现 ---
@utils.register_model # 注册该模型类
class OpenAILLMModel(LLMModel):
    """使用 OpenAI API 的 LLM 模型实现"""

    # setup 方法不再需要处理 _embedding_model
    def setup(self, keys, config): # config 参数在此可能未使用，但为保持签名一致性而保留
        """初始化 OpenAI 客户端"""
        from openai import OpenAI # 导入 OpenAI 库

        # 获取 OpenAI API 密钥，优先从 keys 字典获取，其次从环境变量获取
        api_key = keys.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            # 如果找不到 API 密钥，则抛出错误
            raise ValueError("需要 OpenAI API 密钥。请在 'keys' 中提供或设置 OPENAI_API_KEY 环境变量。")

        # 使用 API 密钥和实例中存储的 _base_url 初始化 OpenAI 客户端
        # _base_url 是由基类 LLMModel.__init__ 设置的
        openai_client = OpenAI(api_key=api_key, base_url=self._base_url)

        # 返回初始化好的客户端句柄 (handle)
        return openai_client

    # _embedding 方法现在从基类 LLMModel 继承，
    # 它会使用 self._handle (即 OpenAI 客户端) 和 self._embedding_model

    # _completion 方法是 OpenAI 特有的聊天补全实现
    def _completion(self, prompt, temperature=0.00001):
        """使用 OpenAI chat completions API 生成文本"""
        # 构建 OpenAI API 需要的消息格式
        messages = [{"role": "user", "content": prompt}]
        try:
            # 调用 OpenAI 客户端的 chat.completions.create 方法
            response = self._handle.chat.completions.create(
                model=self._model,        # 使用实例中存储的模型名称 (例如 "gpt-4o-mini")
                messages=messages,        # 传入构建好的消息列表
                temperature=temperature   # 设置温度参数 (控制随机性)
            )
            # --- 在访问响应内容前进行检查 ---
            # 检查 choices 列表是否存在、不为空，并且第一个 choice 包含 message 属性
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                # 如果检查通过，返回消息内容
                return response.choices[0].message.content
            else:
                # 如果响应结构不符合预期，记录警告信息
                print(f"警告: OpenAI completion 响应缺少 choices/message (模型: {self._model})。")
                # 尝试打印原始响应以便调试
                print(f"--- OpenAI 原始响应 ---")
                if hasattr(response, 'model_dump_json'): print(response.model_dump_json(indent=2)) # Pydantic v2
                elif hasattr(response, 'json'): print(response.json(indent=2) if callable(response.json) else response) # Pydantic v1 或类字典对象
                else: print(response) # 其他情况直接打印
                print(f"-------------------------")
                # 返回 None，表示此次补全失败，以便外部重试逻辑处理
                return None
        except Exception as e:
             # 捕获 API 调用过程中可能发生的异常 (例如网络错误、认证错误等)
             print(f"错误: 调用 OpenAI chat completion API 时出错: {e}")
             # 重新抛出异常，以便外部的 completion 方法的重试逻辑能够捕获并处理
             raise e

    @classmethod
    def support_model(cls, model):
        """检查此类是否支持指定的模型名称"""
        # 更新以包含 gpt-4o-mini
        # 如果需要，可以考虑更灵活的检查 (例如, model.startswith("gpt-"))
        return model in ("gpt-4o-mini", "gpt-3.5-turbo", "text-embedding-3-small")

    @classmethod
    def creatable(cls, keys, config):
        """检查是否满足创建此类的条件 (需要 OpenAI API 密钥)"""
        return "OPENAI_API_KEY" in keys

    @classmethod
    def model_style(cls):
        """返回此模型的风格"""
        return ModelStyle.OPEN_AI

# --- OllamaLLMModel ---
# (保持不变)
@utils.register_model
class OllamaLLMModel(LLMModel):
    def setup(self, keys, config):
        if not self._base_url:
             raise ValueError("base_url is required for OllamaLLMModel.")
        return "OllamaActive"

    def _embedding(self, text):
        headers = {"Content-Type": "application/json"}
        params = {"model": self._embedding_model, "prompt": text}
        try:
            response = requests.post(
                url=f"{self._base_url}/api/embeddings",
                headers=headers,
                json=params,
            )
            response.raise_for_status()
            response_data = response.json()
            if "embedding" in response_data:
                 return response_data["embedding"]
            else:
                 print(f"Warning: Ollama embedding response missing 'embedding' key.")
                 print(f"--- Ollama Raw Response ---\n{response_data}\n-------------------------")
                 return None
        except requests.exceptions.RequestException as e:
             print(f"Error during Ollama embedding request: {e}")
             raise e
        except Exception as e:
             print(f"Error processing Ollama embedding response: {e}")
             raise e

    def _completion(self, prompt, temperature=0.00001):
        headers = {"Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        ollama_native_endpoint = "/api/chat"
        target_url = f"{self._base_url}/chat/completions"
        if ollama_native_endpoint in self._base_url:
             target_url = self._base_url

        try:
            response = requests.post(
                url=target_url,
                headers=headers,
                json=params,
            )
            response.raise_for_status()
            response_data = response.json()

            if target_url.endswith("/chat/completions"):
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                     message = response_data["choices"][0].get("message")
                     if message and message.get("content"):
                         return message["content"]
            elif ollama_native_endpoint in target_url:
                 if response_data.get("message") and response_data["message"].get("content"):
                     return response_data["message"]["content"]

            print(f"Warning: Ollama completion response missing expected content.")
            print(f"--- Ollama Raw Response ---\n{response_data}\n-------------------------")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error during Ollama completion request to {target_url}: {e}")
            raise e
        except Exception as e:
            print(f"Error processing Ollama completion response: {e}")
            raise e

    @classmethod
    def support_model(cls, model):
        return True

    @classmethod
    def creatable(cls, keys, config):
        return True

    @classmethod
    def model_style(cls):
        return ModelStyle.OLLAMA

# --- ZhipuAILLMModel ---
# (保持不变)
@utils.register_model
class ZhipuAILLMModel(LLMModel):
    def setup(self, keys, config):
        from zhipuai import ZhipuAI
        api_key = keys.get("ZHIPUAI_API_KEY")
        if not api_key:
             raise ValueError("ZhipuAI API key is required.")
        return ZhipuAI(api_key=api_key)

    def _embedding(self, text):
        try:
            response = self._handle.embeddings.create(model="embedding-2", input=text)
            if response.data and len(response.data) > 0:
                 return response.data[0].embedding
            else:
                 print(f"Warning: ZhipuAI embedding response had no data.")
                 return None
        except Exception as e:
             print(f"Error during ZhipuAI embedding: {e}")
             raise e

    def _completion(self, prompt, temperature=0.00001):
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._handle.chat.completions.create(
                model=self._model, messages=messages, temperature=temperature
            )
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                return response.choices[0].message.content
            else:
                 print(f"Warning: ZhipuAI completion response missing choices/message.")
                 return None
        except Exception as e:
            print(f"Error during ZhipuAI completion: {e}")
            raise e

    @classmethod
    def support_model(cls, model):
        return model in ("glm-4", "glm-3-turbo")

    @classmethod
    def creatable(cls, keys, config):
        return "ZHIPUAI_API_KEY" in keys

    @classmethod
    def model_style(cls):
        return ModelStyle.ZHIPU_AI

# --- QIANFANLLMModel ---
# (保持不变)
@utils.register_model
class QIANFANLLMModel(LLMModel):
    def setup(self, keys, config):
        needed_keys = ["QIANFAN_AK", "QIANFAN_SK"]
        if not all(k in keys for k in needed_keys):
            raise ValueError("QIANFAN_AK and QIANFAN_SK are required.")
        handle = {k: keys[k] for k in needed_keys}
        for k, v in handle.items():
            os.environ[k] = v
        return handle

    def _get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(
            self._handle["QIANFAN_AK"], self._handle["QIANFAN_SK"]
        )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(""))
            response.raise_for_status()
            response_data = response.json()
            access_token = response_data.get("access_token")
            if not access_token:
                 print("Error: Could not retrieve Qianfan access token.")
                 print(f"--- Qianfan Token Response ---\n{response_data}\n----------------------------")
                 return None
            return access_token
        except requests.exceptions.RequestException as e:
            print(f"Error getting Qianfan access token: {e}")
            return None

    def _embedding(self, text):
        access_token = self._get_access_token()
        if not access_token:
            raise ValueError("Failed to get Qianfan access token for embedding.")

        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token="
            + access_token
        )
        payload = json.dumps({"input": [text]}, ensure_ascii=False)
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("data") and len(response_data["data"]) > 0 and response_data["data"][0].get("embedding"):
                return response_data["data"][0]["embedding"]
            else:
                 print(f"Warning: Qianfan embedding response missing data/embedding.")
                 print(f"--- Qianfan Embedding Response ---\n{response_data}\n-------------------------------")
                 return None
        except requests.exceptions.RequestException as e:
            print(f"Error during Qianfan embedding request: {e}")
            raise e
        except Exception as e:
            print(f"Error processing Qianfan embedding response: {e}")
            raise e

    def _completion(self, prompt, temperature=0.00001):
        try:
             import qianfan
        except ImportError:
             print("Error: 'qianfan' library not installed. Please install it: pip install qianfan")
             raise ImportError("Qianfan library not found")

        messages = [{"role": "user", "content": prompt}]
        try:
             resp = qianfan.ChatCompletion().do(
                 messages=messages, model=self._model, temperature=temperature, disable_search=False
             )
             if resp and resp.get("result"):
                 return resp["result"]
             else:
                  print(f"Warning: Qianfan completion response missing 'result'.")
                  print(f"--- Qianfan Completion Response ---\n{resp}\n--------------------------------")
                  return None
        except Exception as e:
            print(f"Error during Qianfan completion call: {e}")
            raise e

    @classmethod
    def support_model(cls, model):
        return model in ("ERNIE-Bot-4", "ERNIE-Bot-8k", "ERNIE-Bot-turbo", "Yi-34B-Chat", "ERNIE-Speed-128k")

    @classmethod
    def creatable(cls, keys, config):
        return "QIANFAN_AK" in keys and "QIANFAN_SK" in keys

    @classmethod
    def model_style(cls):
        return ModelStyle.QIANFAN

# --- SparkAILLMModel ---
# (保持不变)
@utils.register_model
class SparkAILLMModel(LLMModel):
    def setup(self, keys, config):
        needed_keys = ["SPARK_APPID", "SPARK_API_SECRET", "SPARK_API_KEY"]
        if not all(k in keys for k in needed_keys):
             raise ValueError(f"Missing one or more SparkAI keys: {needed_keys}")

        handle = {"params": {}, "keys": {k: keys[k] for k in needed_keys}}
        spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"

        version_map = {
            "spark_v1.5": ("general", "v1.1"),
            "spark_v2.0": ("generalv2", "v2.1"),
            "spark_v3.0": ("generalv3", "v3.1"),
            "spark_v3.5": ("generalv3.5", "v3.5"),
        }

        if self._model in version_map:
            domain, version_url = version_map[self._model]
            handle["params"] = {
                "domain": domain,
                "spark_url": spark_url_tpl.format(version_url),
            }
        else:
            print(f"Warning: Spark model '{self._model}' not explicitly mapped. Defaulting to v3.5 settings.")
            handle["params"] = {
                "domain": "generalv3.5",
                "spark_url": spark_url_tpl.format("v3.5"),
            }

        handle["keys"] = {k: keys[k] for k in needed_keys}
        return handle

    def _embedding(self, text):
        print("Warning: SparkAI does not provide a direct embedding API through this SDK. Embedding called but not supported.")
        raise NotImplementedError("SparkAI embedding not supported via this implementation.")

    def _completion(self, prompt, temperature=0.5, streaming=False):
        try:
             from sparkai.llm.llm import ChatSparkLLM
             from sparkai.core.messages import ChatMessage
        except ImportError:
             print("Error: 'sparkai' library not installed. Please install it: pip install sparkai")
             raise ImportError("SparkAI library not found")

        spark_llm = ChatSparkLLM(
            spark_api_url=self._handle["params"]["spark_url"],
            spark_app_id=self._handle["keys"]["SPARK_APPID"],
            spark_api_key=self._handle["keys"]["SPARK_API_KEY"],
            spark_api_secret=self._handle["keys"]["SPARK_API_SECRET"],
            spark_llm_domain=self._handle["params"]["domain"],
            temperature=temperature,
            streaming=streaming,
        )
        messages = [ChatMessage(role="user", content=prompt)]
        try:
            resp = spark_llm.generate([messages])
            if resp.generations and len(resp.generations) > 0 and resp.generations[0].message:
                return resp.generations[0].message.content
            else:
                 print(f"Warning: SparkAI completion response missing generations/message.")
                 print(f"--- SparkAI Completion Response ---\n{resp}\n--------------------------------")
                 return None
        except Exception as e:
            print(f"Error during SparkAI completion call: {e}")
            raise e

    @classmethod
    def support_model(cls, model):
        return model in ("spark_v1.5", "spark_v2.0", "spark_v3.0", "spark_v3.5")

    @classmethod
    def creatable(cls, keys, config):
        needed_keys = ["SPARK_APPID", "SPARK_API_SECRET", "SPARK_API_KEY"]
        return all(k in keys for k in needed_keys)

    @classmethod
    def model_style(cls):
        return ModelStyle.SPARK_AI

# --- Factory Function ---
# (保持不变)
def create_llm_model(base_url, model, embedding_model, keys, config=None):
    """创建 llm 模型实例的工厂函数"""
    selected_model_cls = None
    # 遍历所有已注册的 LLM 模型类
    for _, model_cls in utils.get_registered_model(ModelType.LLM).items():
        # 检查模型类是否支持指定的模型名称，并且是否满足创建条件 (例如有对应的 API Key)
        if model_cls.support_model(model) and model_cls.creatable(keys, config):
            selected_model_cls = model_cls # 找到第一个匹配的类
            break # 找到后即退出循环

    if selected_model_cls:
        # 如果找到了合适的模型类
        print(f"找到匹配的模型类: {selected_model_cls.__name__} (模型: {model})，开始创建实例...")
        try:
             # 实例化选中的模型类
             return selected_model_cls(base_url, model, embedding_model, keys, config=config)
        except Exception as e:
             # 捕获实例化过程中可能出现的错误
             print(f"错误: 实例化 {selected_model_cls.__name__} 时出错: {e}")
             raise e # 重新抛出实例化错误
    else:
         # 如果没有找到合适的模型类
         print(f"错误: 找不到适用于模型 '{model}' 且满足所提供密钥条件的已注册 LLM 模型类。")
         # 抛出错误或根据需要返回 None
         raise ValueError(f"不支持的模型或缺少模型所需的密钥: {model}")

# --- Output Parser ---
# (保持不变)
def parse_llm_output(response, patterns, mode="match_last", ignore_empty=False):
    if response is None:
        print("警告: parse_llm_output 接收到的响应为 None。")
        return None if mode != "match_all" else []

    if isinstance(patterns, str):
        patterns = [patterns]
    rets = []
    for line in str(response).split("\n"):
        line = line.replace("**", "").strip()
        if not line and ignore_empty:
             continue
        matched_line = False
        for pattern in patterns:
            try:
                if pattern:
                    matchs = re.findall(pattern, line)
                else:
                    matchs = [line]

                if len(matchs) >= 1:
                    rets.append(matchs[0])
                    matched_line = True
                    break
            except re.error as e:
                print(f"Regex 错误 (parse_llm_output, pattern='{pattern}'): {e}")
                continue

    if not rets and not ignore_empty:
        if not str(response).strip():
             print("警告: LLM 输出为空或仅包含空白字符。")
             return None if mode != "match_all" else []
        else:
             print(f"错误: 无法使用指定模式匹配 LLM 输出: {patterns}")
             print(f"--- LLM 原始输出 ---\n{response}\n----------------------")
             raise ValueError(f"无法匹配 LLM 输出，模式: {patterns}")

    if not rets:
        return None if mode != "match_all" else []

    if mode == "match_first":
        return rets[0]
    if mode == "match_last":
        return rets[-1]
    if mode == "match_all":
        return rets
    return None