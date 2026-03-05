"""
基于LangChain Checkpointer的多对话管理器
提供对话创建、删除、历史记录管理和持久化功能
"""

import os
import uuid
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage


class FileSaver:
    """基于文件的检查点存储器（持久化）"""
    
    def __init__(self, storage_path: str = "./dialog_checkpoints"):
        self.storage_path = os.path.abspath(storage_path)
        print(f"[FileSaver] 存储路径: {self.storage_path}")
        os.makedirs(self.storage_path, exist_ok=True)
        self.store_file = os.path.join(self.storage_path, "checkpoints.json")
        print(f"[FileSaver] 文件路径: {self.store_file}")
        self.store = {}
        self._load_from_file()
    
    def _load_from_file(self):
        """从文件加载数据到内存"""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, 'r', encoding='utf-8') as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}
    
    def _save_to_file(self):
        """将内存数据保存到文件"""
        try:
            serialized = self._serialize_store(self.store)
            print(f"[FileSaver] 准备保存数据到: {self.store_file}")
            print(f"[FileSaver] 当前 store keys: {list(self.store.keys())}")
            with open(self.store_file, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)
            print(f"[FileSaver] 文件保存完成")
            
            # 验证保存成功
            if os.path.exists(self.store_file):
                file_size = os.path.getsize(self.store_file)
                print(f"[FileSaver] 文件存在, 大小: {file_size} bytes")
            else:
                print(f"[FileSaver] 文件不存在!")
        except Exception as e:
            print(f"保存检查点失败: {e}")
    
    def _serialize_store(self, store):
        """递归序列化存储数据，处理 LangChain 消息对象"""
        serialized = {}
        for key, value in store.items():
            serialized[key] = self._serialize_checkpoint(value)
        return serialized
    
    def _serialize_checkpoint(self, checkpoint):
        """序列化检查点数据"""
        if not isinstance(checkpoint, dict):
            return str(checkpoint)
        
        serialized = {}
        for k, v in checkpoint.items():
            if k == "messages" and isinstance(v, list):
                serialized[k] = self._serialize_messages(v)
            elif isinstance(v, dict):
                serialized[k] = self._serialize_checkpoint(v)
            else:
                serialized[k] = v
        return serialized
    
    def _serialize_messages(self, messages):
        """序列化消息列表"""
        serialized = []
        for msg in messages:
            if hasattr(msg, 'content'):
                serialized.append({
                    "type": getattr(msg, 'type', 'unknown'),
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                serialized.append(msg)
        return serialized
    
    def get(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        return self.store.get(thread_id)
    
    def put(self, config, checkpoint, checkpoint_id, metadata):
        thread_id = config.get("configurable", {}).get("thread_id")
        self.store[thread_id] = checkpoint
        self._save_to_file()
    
    def delete(self, config, checkpoint_id):
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id in self.store:
            del self.store[thread_id]
            self._save_to_file()


class MultiDialogManager:
    """基于LangChain Checkpointer的多对话管理器（本地存储）"""
    
    def __init__(self, storage_path: str = "./dialog_checkpoints"):
        self.storage_path = os.path.abspath(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)
        
        self._metadata_file = os.path.join(self.storage_path, "metadata.json")
        self._dialog_metadata = {}
        self._load_metadata()
        
        self.checkpointer = FileSaver(self.storage_path)
        self._memory_store = {}
        self._active_dialogs = set()
        self._load_active_dialogs()
        self._ensure_default_dialog()

    def _load_metadata(self):
        """从磁盘加载对话元数据"""
        if os.path.exists(self._metadata_file):
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    self._dialog_metadata = json.load(f)
            except Exception:
                self._dialog_metadata = {}
        else:
            self._dialog_metadata = {}

    def _save_metadata(self):
        """保存对话元数据到磁盘"""
        try:
            with open(self._metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._dialog_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")

    def _load_active_dialogs(self):
        """从元数据加载活跃对话列表"""
        self._active_dialogs = set(self._dialog_metadata.keys())

    def _ensure_default_dialog(self):
        """确保至少存在一个默认对话"""
        if not self._active_dialogs:
            self.create_dialog("default")

    def create_dialog(self, dialog_id: Optional[str] = None) -> str:
        """创建新对话，返回对话ID"""
        if dialog_id is None:
            dialog_id = str(uuid.uuid4())
        
        if dialog_id not in self._active_dialogs:
            self._active_dialogs.add(dialog_id)
        
        if dialog_id not in self._dialog_metadata:
            self._dialog_metadata[dialog_id] = {
                "title": f"对话 {dialog_id[:8]}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 0
            }
            self._save_metadata()
        
        return dialog_id

    def delete_dialog(self, dialog_id: str):
        """删除对话，如果是最后一个对话则先创建新对话再删除"""
        if dialog_id not in self._active_dialogs:
            raise ValueError(f"对话 {dialog_id} 不存在")
        
        if len(self._active_dialogs) <= 1:
            self.create_dialog()
        
        try:
            self.checkpointer.delete({"configurable": {"thread_id": dialog_id}}, str(uuid.uuid4()))
        except Exception as e:
            print(f"删除检查点失败: {e}")
        
        self._active_dialogs.discard(dialog_id)
        if dialog_id in self._memory_store:
            del self._memory_store[dialog_id]
        if dialog_id in self._dialog_metadata:
            del self._dialog_metadata[dialog_id]
            self._save_metadata()

    def get_memory(self, dialog_id: str) -> Dict[str, Any]:
        """获取指定对话的记忆"""
        if dialog_id not in self._active_dialogs:
            return None
        
        messages = []
        
        try:
            checkpoint = self.checkpointer.get({"configurable": {"thread_id": dialog_id}})
            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                if "messages" in channel_values:
                    msgs = channel_values["messages"]
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if isinstance(msg, dict):
                                role = msg.get("type", "unknown")
                                content = msg.get("content", "")
                                if role in ["human", "ai"]:
                                    messages.append({"role": role, "content": content})
                                else:
                                    messages.append({"role": "assistant", "content": content})
                            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                                role = msg.type if msg.type in ["human", "ai"] else "assistant"
                                messages.append({"role": role, "content": msg.content})
            
            if messages:
                return {"messages": messages}
        except Exception as e:
            print(f"获取对话 {dialog_id} 检查点失败: {e}")
        
        fallback = self._memory_store.get(dialog_id)
        if fallback and isinstance(fallback, dict) and "messages" in fallback:
            return fallback
        
        return {"messages": []}

    def list_dialogs(self) -> List[str]:
        """列出所有活跃对话ID"""
        return list(self._active_dialogs)

    def get_dialog_info(self, dialog_id: str) -> Dict:
        """获取对话的元信息"""
        return self._dialog_metadata.get(dialog_id, {
            "title": f"对话 {dialog_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        })

    def update_dialog_title(self, dialog_id: str, title: str):
        """更新对话标题"""
        if dialog_id not in self._active_dialogs:
            raise ValueError(f"对话 {dialog_id} 不存在")
        
        if dialog_id not in self._dialog_metadata:
            self._dialog_metadata[dialog_id] = {
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 0
            }
        else:
            self._dialog_metadata[dialog_id]["title"] = title
            self._dialog_metadata[dialog_id]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

    def save_context(self, dialog_id: str, human_input: str, ai_output: str):
        """保存对话上下文"""
        checkpoint_id = str(uuid.uuid4())
        
        try:
            current_checkpoint = self.checkpointer.get({"configurable": {"thread_id": dialog_id}})
            current_messages = []
            if current_checkpoint and "channel_values" in current_checkpoint:
                channel_values = current_checkpoint["channel_values"]
                if "messages" in channel_values:
                    msgs = channel_values["messages"]
                    if isinstance(msgs, list):
                        for msg in msgs:
                            current_messages.append(msg)
            
            new_messages = current_messages + [HumanMessage(content=human_input), AIMessage(content=ai_output)]
            
            self.checkpointer.put(
                {"configurable": {"thread_id": dialog_id}},
                {"channel_values": {"messages": new_messages}},
                checkpoint_id,
                {}
            )
            
            if dialog_id in self._dialog_metadata:
                self._dialog_metadata[dialog_id]["message_count"] += 1
                self._dialog_metadata[dialog_id]["updated_at"] = datetime.now().isoformat()
                self._save_metadata()
        except Exception as e:
            print(f"保存对话 {dialog_id} 上下文失败: {e}")

    def increment_message_count(self, dialog_id: str):
        """增加对话消息计数"""
        if dialog_id in self._dialog_metadata:
            self._dialog_metadata[dialog_id]["message_count"] += 1
            self._dialog_metadata[dialog_id]["updated_at"] = datetime.now().isoformat()
            self._save_metadata()


dialog_manager = MultiDialogManager()


def get_dialog_memory_dependency(dialog_id: str = "default"):
    """用于FastAPI依赖注入：获取指定对话记忆（若不存在抛出异常）"""
    mem = dialog_manager.get_memory(dialog_id)
    if mem is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="对话不存在")
    return mem
