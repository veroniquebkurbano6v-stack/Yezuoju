import React, { useEffect, useState } from "react";

export default function ChatWindow({ dialogId, selectedPdf, setSelectedPdf, onCitationsUpdate }) {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pdfs, setPdfs] = useState([]);

  useEffect(() => {
    async function loadHistory() {
      if (!dialogId) {
        setHistory([]);
        return;
      }
      try {
        const res = await fetch(`/api/dialogs/${dialogId}/history`);
        if (!res.ok) {
          throw new Error("对话不存在");
        }
        const data = await res.json();
        const messages = data.history || data.messages || [];
        if (Array.isArray(messages)) {
          setHistory(messages.map(m => ({
            role: m.role,
            text: m.content
          })));
        }
      } catch (e) {
        console.error("加载历史记录失败:", e);
        setHistory([]);
      }
    }
    loadHistory();
  }, [dialogId]);

  useEffect(() => {
    async function fetchPdfs() {
      try {
        const res = await fetch("/api/pdfs");
        const data = await res.json();
        setPdfs(data);
      } catch (err) {
        console.error("获取PDF列表失败:", err);
      }
    }
    fetchPdfs();
  }, []);

  const allPdfFiles = pdfs.flatMap(group => 
    group.files.map(file => ({
      display: `[${group.language}] ${file}`,
      value: file
    }))
  );

  async function send() {
    if (!dialogId) {
      alert("请先在左侧创建或选择一个对话");
      return;
    }
    if (!input.trim()) return;
    setLoading(true);
    try {
      let finalQuestion = input;
      if (selectedPdf) {
        finalQuestion = `${input} [仅在文件：${selectedPdf}]`;
      }
      
      const res = await fetch(`/api/dialogs/${dialogId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: finalQuestion })
      });
      const data = await res.json();
      if (data.success) {
        setHistory((h) => [...h, { role: "user", text: input }, { role: "assistant", text: data.answer }]);
        setInput("");
        // 更新引用展示
        if (onCitationsUpdate && data.retrieved_docs) {
          onCitationsUpdate(data.retrieved_docs);
        }
      } else {
        setHistory((h) => [...h, { role: "assistant", text: "错误：" + (data.error || "未知") }]);
        // 清空引用
        if (onCitationsUpdate) {
          onCitationsUpdate([]);
        }
      }
    } catch (e) {
      setHistory((h) => [...h, { role: "assistant", text: "请求失败" }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-96 bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
      <div className="flex-1 overflow-auto p-4 space-y-3 bg-gradient-to-br from-gray-50 to-gray-100">
        {history.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="w-16 h-16 mb-4 rounded-full bg-indigo-100 flex items-center justify-center">
              <svg className="w-8 h-8 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <p className="text-sm">开始新的对话</p>
            <p className="text-xs mt-1">在下方输入消息开始聊天</p>
          </div>
        )}
        {history.map((m, i) => {
          const isUser = m.role === "user" || m.role === "human";
          return (
          <div
            key={i}
            className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
          >
            {/* 头像 */}
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
              isUser 
                ? "bg-gradient-to-br from-indigo-500 to-purple-600" 
                : "bg-gradient-to-br from-emerald-500 to-teal-600"
            }`}>
              {isUser ? (
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              ) : (
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              )}
            </div>
            
            {/* 消息气泡 */}
            <div className={`max-w-[70%] rounded-2xl px-4 py-3 shadow-sm ${
              isUser
                ? "bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-tr-sm"
                : "bg-white text-gray-800 border border-gray-200 rounded-tl-sm"
            }`}>
              <div className="text-sm leading-relaxed whitespace-pre-wrap">{m.text}</div>
            </div>
          </div>
        )})}
      </div>

      <div className="p-4 bg-white border-t border-gray-100">
        <div className="flex gap-3">
          <select 
            value={selectedPdf} 
            onChange={(e) => setSelectedPdf(e.target.value)}
            className="w-48 border border-gray-200 rounded-xl px-3 py-3 bg-gray-50 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-300"
          >
            <option value="">全部 PDF</option>
            {allPdfFiles.map((item, i) => (
              <option key={i} value={item.value}>{item.display}</option>
            ))}
          </select>
          <input
            className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-300 bg-gray-50 hover:bg-white"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入消息..."
            onKeyDown={(e) => e.key === "Enter" && !loading && input.trim() && send()}
          />
          <button
            className={`px-6 py-3 rounded-xl font-medium text-sm transition-all duration-300 ${
              loading || !input.trim()
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0"
            }`}
            onClick={send}
            disabled={loading || !input.trim()}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                发送中
              </span>
            ) : (
              "发送"
            )}
          </button>
        </div>
        {selectedPdf && (
          <div className="mt-2 text-xs text-gray-400 text-center flex items-center justify-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            已选择: {selectedPdf}
          </div>
        )}
      </div>
    </div>
  );
}
