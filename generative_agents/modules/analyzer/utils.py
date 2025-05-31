# -*- coding: utf-8 -*-
"""
分析器工具模块

包含分析器使用的辅助函数和类。
"""

import json
import os
import traceback
from typing import Dict, Any


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
        print(f"加载agent {agent_name} 失败: {e}")
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        return None