import React, { useEffect, useState } from "react";

export default function DialogsList({ activeDialog, setActiveDialog }) {
  const [dialogs, setDialogs] = useState([]);
  const [dialogInfo, setDialogInfo] = useState({});
  const [loading, setLoading] = useState(false);

  async function fetchDialogs() {
    try {
      const res = await fetch("/api/chat/conversations");
      if (res.ok) {
        const data = await res.json();
        const serverConversations = data.conversations || [];
        const currentLocalDialog = localStorage.getItem("activeDialog");
        const serverIds = serverConversations.map(c => c.session_id);
        const merged = [...new Set([...serverIds, currentLocalDialog].filter(Boolean))];
        setDialogs(merged);
        const infoMap = {};
        serverConversations.forEach(c => {
          infoMap[c.session_id] = {
            title: `对话 ${c.session_id.slice(0, 8)}`,
            message_count: c.message_count || 0,
          };
        });
        merged.forEach(id => {
          if (!infoMap[id]) {
            infoMap[id] = { title: `对话 ${id.slice(0, 8)}`, message_count: 0 };
          }
        });
        setDialogInfo(infoMap);
      }
    } catch (e) {
      console.error("Failed to fetch dialogs:", e);
      const currentLocalDialog = localStorage.getItem("activeDialog");
      if (currentLocalDialog) {
        setDialogs([currentLocalDialog]);
        setDialogInfo({ [currentLocalDialog]: { title: `对话 ${currentLocalDialog.slice(0, 8)}`, message_count: 0 } });
      } else {
        setDialogs([]);
        setDialogInfo({});
      }
    }
  }

  async function createDialog() {
    if (loading) return;
    setLoading(true);
    try {
      const newId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      
      setDialogs(prev => [...prev, newId]);
      setDialogInfo(prev => ({
        ...prev,
        [newId]: { title: `对话 ${newId.slice(0, 8)}`, message_count: 0 }
      }));
      
      setActiveDialog(newId);
      localStorage.setItem("activeDialog", newId);
      
      fetch(`/api/chat/history?conversation_id=${newId}`).catch(() => {});
    } catch (e) {
      console.error("创建对话失败:", e);
    } finally {
      setLoading(false);
    }
  }

  async function deleteDialog(id) {
    try {
      const res = await fetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: id })
      });
      if (!res.ok) {
        console.warn("后端删除对话失败，仅删除本地记录");
      }
      setDialogs(prev => prev.filter(dialogId => dialogId !== id));
      const newInfo = { ...dialogInfo };
      delete newInfo[id];
      setDialogInfo(newInfo);
      
      if (activeDialog === id) {
        setActiveDialog(null);
        localStorage.removeItem("activeDialog");
      }
    } catch (e) {
      console.error("删除对话失败:", e);
    }
  }

  useEffect(() => {
    fetchDialogs();
  }, []);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">对话</h2>
        <button
          onClick={createDialog}
          disabled={loading}
          className={`px-4 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
            loading
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-lg hover:-translate-y-0.5"
          }`}
        >
          {loading ? "创建中..." : "新建对话"}
        </button>
      </div>

      <div className="space-y-2">
        {dialogs.length === 0 && (
          <div className="text-sm text-gray-400 text-center py-8 bg-gray-50 rounded-lg">
暂无对话，点击"新建对话"开始
          </div>
        )}
        {dialogs.map((id) => {
          const info = dialogInfo[id] || {};
          const title = info.title || `对话 ${id.slice(0, 8)}`;
          const messageCount = info.message_count || 0;
          return (
          <div
            key={id}
            className={`p-3 rounded-lg border transition-all duration-300 cursor-pointer ${
              activeDialog === id
                ? "border-indigo-400 bg-indigo-50 shadow-sm"
                : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/50 hover:shadow-md"
            }`}
          >
            <div className="flex justify-between items-center">
              <button
                className="text-left text-gray-700 hover:text-indigo-600 transition-colors duration-200 font-medium flex-1"
                onClick={() => setActiveDialog(id)}
              >
                <div className="text-sm font-semibold">{title}</div>
                <div className="text-xs text-gray-500 mt-1">{messageCount} 条消息</div>
              </button>
              <button
                onClick={() => deleteDialog(id)}
                className="text-red-500 text-sm hover:text-red-700 hover:bg-red-50 px-2 py-1 rounded transition-all duration-200"
              >
                删除
              </button>
            </div>
          </div>
        );})}
      </div>
    </div>
  );
}


