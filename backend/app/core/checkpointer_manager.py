import os
import uuid
import json
from typing import Optional, Dict, List
from datetime import datetime

try:
    from langchain.memory import ConversationTokenBufferMemory
    from langchain_community.checkpoint.agent_memory import SimpleCheckpointSaver
    from langchain_core.chat_history import InMemoryChatMessageHistory
except Exception:
    ConversationTokenBufferMemory = None
    SimpleCheckpointSaver = None
    InMemoryChatMessageHistory = None


class MultiDialogManager:
    """基于Checkpointer的多对话管理器（本地存储）"""
    def __init__(self, storage_path: str = "./dialog_checkpoints", llm=None):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._active_dialogs = {}
        self._dialog_metadata = {}
        self.llm = llm
        self._load_all_dialogs()
        self._ensure_default_dialog()

    def _load_all_dialogs(self):
        """从磁盘加载所有对话的元数据和消息内容"""
        metadata_file = os.path.join(self.storage_path, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self._dialog_metadata = json.load(f)
            except Exception:
                self._dialog_metadata = {}
        
        for dialog_id in self._dialog_metadata.keys():
            messages_file = os.path.join(self.storage_path, f"{dialog_id}.json")
            if os.path.exists(messages_file):
                try:
                    with open(messages_file, 'r', encoding='utf-8') as f:
                        messages_data = json.load(f)
                        self._active_dialogs[dialog_id] = {"messages": messages_data.get("messages", [])}
                except Exception:
                    self._active_dialogs[dialog_id] = {"messages": []}
            else:
                self._active_dialogs[dialog_id] = {"messages": []}

    def _serialize_message_content(self, content):
        """序列化消息内容，处理不可序列化的对象"""
        if isinstance(content, str):
            return content
        elif hasattr(content, '__dict__'):
            # 如果是可序列化对象，转换为字典
            return str(content)
        else:
            return str(content)

    def _save_messages(self, dialog_id: str):
        """保存指定对话的消息内容到磁盘"""
        memory = self._active_dialogs.get(dialog_id)
        if memory is None:
            return
        
        messages_file = os.path.join(self.storage_path, f"{dialog_id}.json")
        try:
            if isinstance(memory, dict) and "messages" in memory:
                # 处理消息内容，确保可序列化
                serialized_messages = []
                for msg in memory["messages"]:
                    serialized_msg = {
                        "role": msg.get("role", "unknown"),
                        "content": self._serialize_message_content(msg.get("content", "")),
                    }
                    # 如果有 references，先转成字符串
                    if "references" in msg and msg["references"]:
                        try:
                            serialized_msg["content"] = str(msg["content"]) + f"\n\n引用: {len(msg['references'])} 个来源"
                        except:
                            serialized_msg["content"] = str(msg["content"])
                    serialized_messages.append(serialized_msg)
                messages_data = {"messages": serialized_messages}
            elif hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                messages_data = {"messages": [
                    {"role": msg.type if hasattr(msg, 'type') else "unknown", "content": self._serialize_message_content(msg.content if hasattr(msg, 'content') else str(msg))}
                    for msg in memory.chat_memory.messages
                ]}
            else:
                return
            
            with open(messages_file, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话 {dialog_id} 消息失败: {e}")

    def _load_metadata(self):
        """从磁盘加载对话元数据"""
        metadata_file = os.path.join(self.storage_path, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self._dialog_metadata = json.load(f)
            except Exception:
                self._dialog_metadata = {}

    def _save_metadata(self):
        """保存对话元数据到磁盘"""
        metadata_file = os.path.join(self.storage_path, "metadata.json")
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._dialog_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")

    def _ensure_default_dialog(self):
        """确保至少存在一个默认对话"""
        if not self._active_dialogs:
            default_id = "default"
            self.create_dialog(default_id)

    def create_dialog(self, dialog_id: Optional[str] = None) -> str:
        """创建新对话，返回对话ID"""
        if dialog_id is None:
            dialog_id = str(uuid.uuid4())

        messages_file = os.path.join(self.storage_path, f"{dialog_id}.json")
        
        if os.path.exists(messages_file):
            try:
                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages_data = json.load(f)
                    self._active_dialogs[dialog_id] = {"messages": messages_data.get("messages", [])}
            except Exception:
                self._active_dialogs[dialog_id] = {"messages": []}
        else:
            self._active_dialogs[dialog_id] = {"messages": []}
        
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
        
        # 如果只剩一个对话，先创建一个新对话
        if len(self._active_dialogs) <= 1:
            self.create_dialog()
        
        # 删除对话文件和内存记录
        checkpoint_file = os.path.join(self.storage_path, f"{dialog_id}.json")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        del self._active_dialogs[dialog_id]
        if dialog_id in self._dialog_metadata:
            del self._dialog_metadata[dialog_id]
        self._save_metadata()

    def get_memory(self, dialog_id: str):
        """获取指定对话的记忆"""
        return self._active_dialogs.get(dialog_id)

    def list_dialogs(self) -> list:
        """列出所有活跃对话ID"""
        return list(self._active_dialogs.keys())

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

    def increment_message_count(self, dialog_id: str):
        """增加对话消息计数"""
        if dialog_id in self._dialog_metadata:
            self._dialog_metadata[dialog_id]["message_count"] += 1
            self._dialog_metadata[dialog_id]["updated_at"] = datetime.now().isoformat()
            self._save_metadata()
            self._save_messages(dialog_id)


# 全局实例（上层可在启动时替换 llm）
dialog_manager = MultiDialogManager()


def get_dialog_memory_dependency(dialog_id: str = "default"):
    """用于 FastAPI 依赖注入：获取指定对话记忆（若不存在抛出异常）"""
    mem = dialog_manager.get_memory(dialog_id)
    if mem is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="对话不存在")
    return mem




