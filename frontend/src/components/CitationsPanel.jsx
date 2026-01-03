import React from "react";

export default function CitationsPanel({ citations }) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden mt-4">
      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-white text-lg">📚</span>
          <h3 className="text-sm font-semibold text-white">引用来源（共 {citations.length} 条）</h3>
        </div>
      </div>
      
      <div className="max-h-80 overflow-y-auto custom-scrollbar">
        {citations.map((item, index) => (
          <div 
            key={index}
            className="p-4 border-b border-gray-100 hover:bg-indigo-50/50 transition-colors duration-200"
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center">
                <span className="text-xs font-medium text-indigo-600">{index + 1}</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <span className="inline-flex items-center px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-md font-medium">
                    📄 {item.pdf_filename}
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-green-50 text-green-700 text-xs rounded-md font-medium">
                    📑 {item.section_title}
                  </span>
                  <span className="inline-flex items-center px-2 py-1 bg-orange-50 text-orange-700 text-xs rounded-md font-medium">
                    📍 第 {item.page_number} 页
                  </span>
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min((item.score || 0) * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 font-medium flex-shrink-0">
                    {item.score ? (item.score * 100).toFixed(1) : 0}%
                  </span>
                </div>
                {item.text && (
                  <div className="pl-2 border-l-2 border-indigo-200">
                    <p className="text-xs text-gray-600 leading-relaxed line-clamp-3 italic">
                      "{item.text}"
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-50 px-4 py-2 border-t border-gray-100">
        <p className="text-xs text-gray-400 text-center">
          💡 点击对话可查看完整引用来源
        </p>
      </div>
    </div>
  );
}
