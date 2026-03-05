import React, { useEffect, useState } from "react";

export default function DialogsList({ activeDialog, setActiveDialog }) {
  const [dialogs, setDialogs] = useState([]);
  const [dialogInfo, setDialogInfo] = useState({});
  const [loading, setLoading] = useState(false);

  async function fetchDialogs() {
    try {
      const res = await fetch("/api/dialogs/");
      if (res.ok) {
        const data = await res.json();
        setDialogs(data.dialogs || []);
        // 构建 dialog_info 映射
        if (data.dialog_info && Array.isArray(data.dialog_info)) {
          const infoMap = {};
          data.dialog_info.forEach((info, idx) => {
            if (data.dialogs && data.dialogs[idx]) {
              infoMap[data.dialogs[idx]] = info;
            }
          });
          setDialogInfo(infoMap);
        }
      }
    } catch (e) {
      console.error("Failed to fetch dialogs:", e);
      setDialogs([]);
      setDialogInfo({});
    }
  }

  async function createDialog() {
    if (loading) return;
    setLoading(true);
    try {
      const res = await fetch("/api/dialogs/", { method: "POST" });
      const data = await res.json();
      if (data.dialog_id) {
        setActiveDialog(data.dialog_id);
        await fetchDialogs();
      }
    } catch (e) {
      console.error("创建对话失败:", e);
    } finally {
      setLoading(false);
    }
  }

  async function deleteDialog(id) {
    try {
      const res = await fetch(`/api/dialogs/${id}`, { method: "DELETE" });
      if (res.ok) {
        await fetchDialogs();
        if (activeDialog === id) {
          setActiveDialog(null);
          localStorage.removeItem("activeDialog");
        }
      }
    } catch (e) {
      console.error(e);
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


