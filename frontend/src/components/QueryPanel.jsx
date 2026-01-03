import React, { useState, useEffect } from "react";
import ResultCard from "./ResultCard";

export default function QueryPanel({ selectedPdf, setSelectedPdf }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pdfs, setPdfs] = useState([]);

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

  async function runQuery(e) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    try {
      // 如果选择了PDF，在查询中附加文件名
      let finalQuery = q;
      if (selectedPdf) {
        finalQuery = `${q} [仅在文件：${selectedPdf}]`;
      }
      
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: finalQuery })
      });
      const data = await res.json();
      setResults(data.references || []);
    } catch (err) {
      console.error(err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  const allPdfFiles = pdfs.flatMap(group => 
    group.files.map(file => ({
      display: `[${group.language}] ${file}`,
      value: file
    }))
  );

  return (
    <div className="mt-4 space-y-3 relative z-10">
      {results.length === 0 && (
        <div className="flex flex-col items-center justify-center py-8">
          <div className="flex items-center gap-2 text-indigo-300/80">
            <span className="text-lg">🌟</span>
            <span className="text-lg font-serif tracking-wide">请在下方对话框中输入问题进行对话</span>
            <span className="text-lg">✨</span>
          </div>
        </div>
      )}
      {results.map((r, i) => <ResultCard key={r.chunk_id || i} result={r} index={i+1} />)}
    </div>
  );
}




