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
        const data = await res.json();
        if (data.history) {
          setHistory(data.history.map(m => ({
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
    <div className="flex flex-col h-96 bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div className="flex-1 overflow-auto p-4 space-y-4 bg-gray-50/50">
        {history.length === 0 && (
          <div className="text-sm text-gray-400 text-center py-12">
            <div className="mb-2">💬</div>
            对话尚未开始<br />
            请在左侧创建或选择一个对话，然后发送消息
          </div>
        )}
        {history.map((m, i) => (
          <div
            key={i}
            className={`max-w-[85%] p-4 rounded-2xl transition-all duration-300 whitespace-pre-wrap ${
              m.role === "user"
                ? "bg-indigo-600 text-white ml-auto shadow-md hover:shadow-lg"
                : "bg-white text-gray-700 mr-auto shadow-sm border border-gray-100 hover:shadow-md"
            }`}
          >
            <div className="text-sm leading-relaxed">{m.text}</div>
          </div>
        ))}
      </div>

      <div className="p-4 bg-white border-t border-gray-100">
        <div className="flex gap-3">
          <select 
            value={selectedPdf} 
            onChange={(e) => setSelectedPdf(e.target.value)}
            className="w-48 border border-gray-200 rounded-xl px-3 py-3 bg-gray-50 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all duration-300"
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
                : "bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0"
            }`}
            onClick={send}
            disabled={loading || !input.trim()}
          >
            {loading ? "发送中..." : "发送"}
          </button>
        </div>
        {selectedPdf && (
          <div className="mt-2 text-xs text-gray-400 text-center">
            已选择 PDF: {selectedPdf}
          </div>
        )}
      </div>
    </div>
  );
}
