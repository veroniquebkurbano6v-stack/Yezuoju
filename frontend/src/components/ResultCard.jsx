import React from "react";

export default function ResultCard({ result, index }) {
  return (
    <div className="border rounded p-3 bg-white">
      <div className="flex justify-between items-start">
        <div>
          <div className="text-sm text-gray-500">[{index}] {result.pdf_filename}</div>
          <div className="font-medium">{result.section_title || "未知章节"}</div>
          <div className="text-xs text-gray-400">页码: {result.page_number}</div>
        </div>
        <div className="text-sm text-indigo-600">{(result.score || 0).toFixed(3)}</div>
      </div>
      <div className="mt-2 text-preview">{result.text.slice(0, 400)}</div>
    </div>
  );
}




