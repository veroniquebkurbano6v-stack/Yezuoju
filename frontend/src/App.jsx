import React, { useEffect, useState, useCallback, useRef } from "react";

import DialogsList from "./components/DialogsList";
import ChatWindow from "./components/ChatWindow";
import QueryPanel from "./components/QueryPanel";
import CitationsPanel from "./components/CitationsPanel";
import Star from "./components/Star";

const starConfigs = [
  { id: 0, color: "pink", size: "w-1.5 h-1.5", top: "5%", left: "8%", delay: "0s", duration: "2.5s", opacity: "80" },
  { id: 1, color: "cyan", size: "w-1 h-1", top: "7%", left: "22%", delay: "0.3s", duration: "3s", opacity: "70" },
  { id: 2, color: "yellow", size: "w-1.5 h-1.5", top: "4%", left: "38%", delay: "0.6s", duration: "2.8s", opacity: "80" },
  { id: 3, color: "purple", size: "w-1 h-1", top: "9%", left: "52%", delay: "0.9s", duration: "3.2s", opacity: "70" },
  { id: 4, color: "green", size: "w-1.5 h-1.5", top: "6%", left: "68%", delay: "1.2s", duration: "2.6s", opacity: "80" },
  { id: 5, color: "pink", size: "w-1 h-1", top: "11%", left: "82%", delay: "1.5s", duration: "3.4s", opacity: "70" },
  { id: 6, color: "cyan", size: "w-1.5 h-1.5", top: "8%", left: "95%", delay: "1.8s", duration: "2.9s", opacity: "80" },
  { id: 7, color: "yellow", size: "w-1 h-1", top: "13%", left: "15%", delay: "2.1s", duration: "3.1s", opacity: "70" },
  { id: 8, color: "purple", size: "w-1.5 h-1.5", top: "10%", left: "30%", delay: "0.2s", duration: "2.7s", opacity: "80" },
  { id: 9, color: "pink", size: "w-1 h-1", top: "15%", left: "45%", delay: "0.5s", duration: "3.3s", opacity: "70" },
  { id: 10, color: "green", size: "w-1.5 h-1.5", top: "12%", left: "60%", delay: "0.8s", duration: "2.4s", opacity: "80" },
  { id: 11, color: "cyan", size: "w-1 h-1", top: "16%", left: "75%", delay: "1.1s", duration: "3.5s", opacity: "70" },
  { id: 12, color: "yellow", size: "w-1.5 h-1.5", top: "14%", left: "88%", delay: "1.4s", duration: "2.5s", opacity: "80" },
  { id: 13, color: "purple", size: "w-1 h-1", top: "18%", left: "5%", delay: "1.7s", duration: "3s", opacity: "70" },
  { id: 14, color: "pink", size: "w-1.5 h-1.5", top: "20%", left: "20%", delay: "2s", duration: "2.8s", opacity: "80" },
  { id: 15, color: "green", size: "w-1 h-1", top: "22%", left: "35%", delay: "0.4s", duration: "3.2s", opacity: "70" },
  { id: 16, color: "cyan", size: "w-1.5 h-1.5", top: "19%", left: "50%", delay: "0.7s", duration: "2.6s", opacity: "80" },
  { id: 17, color: "yellow", size: "w-1 h-1", top: "24%", left: "65%", delay: "1s", duration: "3.4s", opacity: "70" },
  { id: 18, color: "purple", size: "w-1.5 h-1.5", top: "21%", left: "80%", delay: "1.3s", duration: "2.7s", opacity: "80" },
  { id: 19, color: "pink", size: "w-1 h-1", top: "26%", left: "92%", delay: "1.6s", duration: "3.1s", opacity: "70" },
  { id: 20, color: "green", size: "w-1.5 h-1.5", top: "28%", left: "12%", delay: "1.9s", duration: "2.9s", opacity: "80" },
  { id: 21, color: "cyan", size: "w-1 h-1", top: "30%", left: "28%", delay: "0.1s", duration: "3.3s", opacity: "70" },
  { id: 22, color: "yellow", size: "w-1.5 h-1.5", top: "27%", left: "42%", delay: "0.4s", duration: "2.5s", opacity: "80" },
  { id: 23, color: "purple", size: "w-1 h-1", top: "32%", left: "55%", delay: "0.7s", duration: "3s", opacity: "70" },
  { id: 24, color: "pink", size: "w-1.5 h-1.5", top: "29%", left: "70%", delay: "1s", duration: "2.8s", opacity: "80" },
  { id: 25, color: "green", size: "w-1 h-1", top: "34%", left: "85%", delay: "1.3s", duration: "3.2s", opacity: "70" },
  { id: 26, color: "cyan", size: "w-1.5 h-1.5", top: "36%", left: "8%", delay: "1.6s", duration: "2.6s", opacity: "80" },
  { id: 27, color: "yellow", size: "w-1 h-1", top: "38%", left: "25%", delay: "1.9s", duration: "3.4s", opacity: "70" },
  { id: 28, color: "purple", size: "w-1.5 h-1.5", top: "35%", left: "40%", delay: "0.2s", duration: "2.7s", opacity: "80" },
  { id: 29, color: "pink", size: "w-1 h-1", top: "40%", left: "58%", delay: "0.5s", duration: "3.1s", opacity: "70" },
  { id: 30, color: "green", size: "w-1.5 h-1.5", top: "37%", left: "72%", delay: "0.8s", duration: "2.9s", opacity: "80" },
  { id: 31, color: "cyan", size: "w-1 h-1", top: "42%", left: "88%", delay: "1.1s", duration: "3.3s", opacity: "70" },
  { id: 32, color: "yellow", size: "w-1.5 h-1.5", top: "44%", left: "3%", delay: "1.4s", duration: "2.5s", opacity: "80" },
  { id: 33, color: "purple", size: "w-1 h-1", top: "46%", left: "18%", delay: "1.7s", duration: "3.2s", opacity: "70" },
  { id: 34, color: "pink", size: "w-1.5 h-1.5", top: "43%", left: "35%", delay: "2s", duration: "2.8s", opacity: "80" },
  { id: 35, color: "green", size: "w-1 h-1", top: "48%", left: "52%", delay: "0.3s", duration: "3s", opacity: "70" },
  { id: 36, color: "cyan", size: "w-1.5 h-1.5", top: "45%", left: "68%", delay: "0.6s", duration: "2.6s", opacity: "80" },
  { id: 37, color: "yellow", size: "w-1 h-1", top: "50%", left: "82%", delay: "0.9s", duration: "3.4s", opacity: "70" },
  { id: 38, color: "purple", size: "w-1.5 h-1.5", top: "52%", left: "95%", delay: "1.2s", duration: "2.7s", opacity: "80" },
  { id: 39, color: "pink", size: "w-1 h-1", top: "54%", left: "10%", delay: "1.5s", duration: "3.1s", opacity: "70" },
  { id: 40, color: "green", size: "w-1.5 h-1.5", top: "51%", left: "28%", delay: "1.8s", duration: "2.9s", opacity: "80" },
  { id: 41, color: "cyan", size: "w-1 h-1", top: "56%", left: "45%", delay: "0.1s", duration: "3.3s", opacity: "70" },
  { id: 42, color: "yellow", size: "w-1.5 h-1.5", top: "53%", left: "62%", delay: "0.4s", duration: "2.5s", opacity: "80" },
  { id: 43, color: "purple", size: "w-1 h-1", top: "58%", left: "78%", delay: "0.7s", duration: "3s", opacity: "70" },
  { id: 44, color: "pink", size: "w-1.5 h-1.5", top: "60%", left: "92%", delay: "1s", duration: "2.8s", opacity: "80" },
  { id: 45, color: "green", size: "w-1 h-1", top: "62%", left: "15%", delay: "1.3s", duration: "3.2s", opacity: "70" },
  { id: 46, color: "cyan", size: "w-1.5 h-1.5", top: "59%", left: "32%", delay: "1.6s", duration: "2.6s", opacity: "80" },
  { id: 47, color: "yellow", size: "w-1 h-1", top: "64%", left: "50%", delay: "1.9s", duration: "3.4s", opacity: "70" },
  { id: 48, color: "purple", size: "w-1.5 h-1.5", top: "61%", left: "68%", delay: "0.2s", duration: "2.7s", opacity: "80" },
  { id: 49, color: "pink", size: "w-1 h-1", top: "66%", left: "85%", delay: "0.5s", duration: "3.1s", opacity: "70" },
  { id: 50, color: "green", size: "w-1.5 h-1.5", top: "68%", left: "5%", delay: "0.8s", duration: "2.9s", opacity: "80" },
  { id: 51, color: "cyan", size: "w-1 h-1", top: "70%", left: "22%", delay: "1.1s", duration: "3.3s", opacity: "70" },
  { id: 52, color: "yellow", size: "w-1.5 h-1.5", top: "67%", left: "40%", delay: "1.4s", duration: "2.5s", opacity: "80" },
  { id: 53, color: "purple", size: "w-1 h-1", top: "72%", left: "58%", delay: "1.7s", duration: "3s", opacity: "70" },
  { id: 54, color: "pink", size: "w-1.5 h-1.5", top: "69%", left: "75%", delay: "2s", duration: "2.8s", opacity: "80" },
  { id: 55, color: "green", size: "w-1 h-1", top: "74%", left: "90%", delay: "0.3s", duration: "3.2s", opacity: "70" },
  { id: 56, color: "cyan", size: "w-1.5 h-1.5", top: "76%", left: "12%", delay: "0.6s", duration: "2.6s", opacity: "80" },
  { id: 57, color: "yellow", size: "w-1 h-1", top: "78%", left: "30%", delay: "0.9s", duration: "3.4s", opacity: "70" },
  { id: 58, color: "purple", size: "w-1.5 h-1.5", top: "75%", left: "48%", delay: "1.2s", duration: "2.7s", opacity: "80" },
  { id: 59, color: "pink", size: "w-1 h-1", top: "80%", left: "65%", delay: "1.5s", duration: "3.1s", opacity: "70" },
  { id: 60, color: "green", size: "w-1.5 h-1.5", top: "77%", left: "82%", delay: "1.8s", duration: "2.9s", opacity: "80" },
  { id: 61, color: "cyan", size: "w-1 h-1", top: "82%", left: "95%", delay: "2.1s", duration: "3.3s", opacity: "70" },
  { id: 62, color: "yellow", size: "w-1.5 h-1.5", top: "84%", left: "8%", delay: "0.1s", duration: "2.5s", opacity: "80" },
  { id: 63, color: "purple", size: "w-1 h-1", top: "86%", left: "25%", delay: "0.4s", duration: "3s", opacity: "70" },
  { id: 64, color: "pink", size: "w-1.5 h-1.5", top: "83%", left: "42%", delay: "0.7s", duration: "2.8s", opacity: "80" },
  { id: 65, color: "green", size: "w-1 h-1", top: "88%", left: "60%", delay: "1s", duration: "3.2s", opacity: "70" },
  { id: 66, color: "cyan", size: "w-1.5 h-1.5", top: "85%", left: "78%", delay: "1.3s", duration: "2.6s", opacity: "80" },
  { id: 67, color: "yellow", size: "w-1 h-1", top: "90%", left: "92%", delay: "1.6s", duration: "3.4s", opacity: "70" },
  { id: 68, color: "purple", size: "w-1.5 h-1.5", top: "92%", left: "18%", delay: "1.9s", duration: "2.7s", opacity: "80" },
  { id: 69, color: "pink", size: "w-1 h-1", top: "94%", left: "38%", delay: "0.2s", duration: "3.1s", opacity: "70" },
  { id: 70, color: "green", size: "w-1.5 h-1.5", top: "91%", left: "55%", delay: "0.5s", duration: "2.9s", opacity: "80" },
  { id: 71, color: "cyan", size: "w-1 h-1", top: "96%", left: "72%", delay: "0.8s", duration: "3.3s", opacity: "70" },
  { id: 72, color: "yellow", size: "w-1.5 h-1.5", top: "93%", left: "88%", delay: "1.1s", duration: "2.5s", opacity: "80" },
  { id: 73, color: "orange", size: "w-2 h-2", top: "10%", left: "55%", delay: "0s", duration: "4s", opacity: "90" },
  { id: 74, color: "rose", size: "w-2 h-2", top: "30%", left: "10%", delay: "1s", duration: "5s", opacity: "90" },
  { id: 75, color: "sky", size: "w-2 h-2", top: "50%", left: "85%", delay: "2s", duration: "4.5s", opacity: "90" },
  { id: 76, color: "amber", size: "w-2 h-2", top: "70%", left: "25%", delay: "0.5s", duration: "5.5s", opacity: "90" },
  { id: 77, color: "fuchsia", size: "w-2 h-2", top: "85%", left: "70%", delay: "3s", duration: "4s", opacity: "90" },
];

function shuffleArray(array) {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
}

export default function App() {
  const [dialogs, setDialogs] = useState([]);
  const [activeDialog, setActiveDialog] = useState(null);
  const [selectedPdf, setSelectedPdf] = useState(""); // 全局 PDF 选择状态
  const [citations, setCitations] = useState([]); // 引用展示数据
  const [activeStarIds, setActiveStarIds] = useState([]);
  const [starOrder, setStarOrder] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const timeoutRef = useRef(null);
  const initializedRef = useRef(false);
  const currentIndexRef = useRef(0);
  const starOrderRef = useRef([]);
  const isAnimatingRef = useRef(false);

  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true;
      const shuffledOrder = shuffleArray(starConfigs.map(s => s.id));
      starOrderRef.current = shuffledOrder;
      setStarOrder(shuffledOrder);
      currentIndexRef.current = 0;
      setCurrentIndex(0);

      timeoutRef.current = setTimeout(() => {
        isAnimatingRef.current = true;
        const firstThree = shuffledOrder.slice(0, 3);
        setActiveStarIds(firstThree);
      }, 100);
    }
  }, []);

  useEffect(() => {
    if (!initializedRef.current) return;
    if (starOrder.length === 0) return;
    if (isAnimatingRef.current) return;

    const nextIndex = currentIndexRef.current + 3;
    if (nextIndex >= starOrderRef.current.length) {
      const newOrder = shuffleArray(starConfigs.map(s => s.id));
      starOrderRef.current = newOrder;
      setStarOrder(newOrder);
      currentIndexRef.current = 0;
      setCurrentIndex(0);
      isAnimatingRef.current = true;
      const firstThree = newOrder.slice(0, 3);
      setActiveStarIds(firstThree);
    } else {
      currentIndexRef.current = nextIndex;
      setCurrentIndex(nextIndex);
      isAnimatingRef.current = true;
      const nextThree = starOrderRef.current.slice(nextIndex, nextIndex + 3);
      setActiveStarIds(nextThree);
    }
  }, [currentIndex]);

  const handleStarAnimationComplete = useCallback(() => {
    setActiveStarIds([]);
    isAnimatingRef.current = false;

    timeoutRef.current = setTimeout(() => {
      setCurrentIndex(prev => {
        const nextIndex = prev + 3;
        if (nextIndex >= starOrderRef.current.length) {
          const newOrder = shuffleArray(starConfigs.map(s => s.id));
          starOrderRef.current = newOrder;
          setStarOrder(newOrder);
          currentIndexRef.current = 0;
          isAnimatingRef.current = true;
          setActiveStarIds(newOrder.slice(0, 3));
          return 0;
        }
        return nextIndex;
      });
    }, 3000);
  }, []);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="relative bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 shadow-lg overflow-hidden">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRjMC0yIDItNCAyLTRzLTItMi00LTJjMCAwLTItMi0yLTRzMi00IDItNCAyIDIgNCAyczQgMiA0IDQtMiA0LTIgNC0yIDItNCAyYzAgMC0yIDItMiA0czIgNCAyIDQiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-30"></div>
        
        <div className="absolute top-4 right-20 w-20 h-20 bg-cyan-300 opacity-20 rounded-full blur-xl animate-bounce" style={{animationDuration: '3s'}}></div>
        <div className="absolute top-8 right-40 w-16 h-16 bg-yellow-300 opacity-20 rounded-full blur-xl animate-bounce" style={{animationDuration: '2.5s', animationDelay: '0.5s'}}></div>
        <div className="absolute top-12 right-60 w-12 h-12 bg-pink-300 opacity-20 rounded-full blur-xl animate-bounce" style={{animationDuration: '2s', animationDelay: '1s'}}></div>
        
        <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-5 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-pink-300 opacity-10 rounded-full blur-3xl transform -translate-x-1/2 translate-y-1/2"></div>
        <div className="absolute top-1/2 left-10 w-32 h-32 bg-blue-300 opacity-10 rounded-full blur-2xl animate-pulse"></div>
        
        <div className="absolute top-20 left-1/4 text-white/10 text-6xl animate-spin" style={{animationDuration: '20s'}}>✨</div>
        <div className="absolute bottom-10 right-1/4 text-white/10 text-4xl animate-pulse">📖</div>
        <div className="absolute top-10 right-1/3 text-white/10 text-3xl animate-bounce">🔮</div>
        
        <div className="relative max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-14 h-14 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg">
                <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                  <line x1="12" y1="6" x2="12" y2="10"></line>
                  <line x1="12" y1="14" x2="12" y2="14"></line>
                  <line x1="9" y1="18" x2="15" y2="18"></line>
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-5 h-5 bg-green-400 rounded-full border-2 border-white animate-ping"></div>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white drop-shadow-lg flex items-center gap-3 flex-wrap">
                <span className="bg-gradient-to-r from-yellow-200 via-white to-pink-200 bg-clip-text text-transparent">
                  StoryRag
                </span>
                <span className="text-2xl font-light text-white/70">✨</span>
                <span className="text-xl font-normal text-white/90">智能检索</span>
              </h1>
              <p className="text-sm text-white/80 mt-2 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span className="bg-white/10 px-3 py-1 rounded-full backdrop-blur-sm">
                  搜索您的 PDF 库并用 LLM 生成带引证的答案
                </span>
              </p>
            </div>
          </div>
          
          <div className="mt-6 flex gap-2">
            <div className="px-4 py-1.5 bg-white/10 backdrop-blur-sm rounded-full text-white/80 text-xs flex items-center gap-2">
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></span>
              AI 驱动
            </div>
            <div className="px-4 py-1.5 bg-white/10 backdrop-blur-sm rounded-full text-white/80 text-xs flex items-center gap-2">
              <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></span>
              智能检索
            </div>
            <div className="px-4 py-1.5 bg-white/10 backdrop-blur-sm rounded-full text-white/80 text-xs flex items-center gap-2">
              <span className="w-2 h-2 bg-pink-400 rounded-full animate-pulse" style={{animationDelay: '1s'}}></span>
              多语言支持
            </div>
          </div>
        </div>
        
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-black"></div>
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/10 to-transparent pointer-events-none"></div>
      </header>

      <div className="h-32 bg-gradient-to-b from-indigo-900/30 via-purple-900/50 to-slate-900"></div>

      <main className="relative -mt-32 min-h-screen bg-gradient-to-b from-slate-900 via-purple-900 to-slate-900 overflow-y-auto">
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          {starConfigs.map(star => (
            <Star 
              key={star.id} 
              id={star.id}
              activeIds={activeStarIds}
              color={star.color} 
              size={star.size} 
              top={star.top} 
              left={star.left} 
              delay={star.delay} 
              duration={star.duration} 
              opacity={star.opacity}
              onAnimationComplete={handleStarAnimationComplete}
            />
          ))}
          
          <div className="absolute top-0 left-0 w-80 h-80 bg-indigo-900/40 rounded-full blur-3xl transform -translate-x-1/2 translate-y-1/4"></div>
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-pink-900/40 rounded-full blur-3xl transform translate-x-1/3 -translate-y-1/3"></div>
          <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-purple-900/30 rounded-full blur-3xl transform -translate-x-1/2 -translate-y-1/2"></div>
        </div>
        
        <div className="relative z-10 max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-slate-900/95 backdrop-blur-md rounded-xl border border-white/10 shadow-xl p-1">
              <DialogsList activeDialog={activeDialog} setActiveDialog={(id) => { setActiveDialog(id); localStorage.setItem("activeDialog", id); }} />
            </div>
          </div>

          <div className="lg:col-span-3 space-y-6">
            <div className="bg-slate-900/95 backdrop-blur-md rounded-xl border border-white/10 shadow-xl p-1">
              <QueryPanel selectedPdf={selectedPdf} setSelectedPdf={setSelectedPdf} />
            </div>

            <div className="bg-slate-900/95 backdrop-blur-md rounded-xl border border-white/10 shadow-xl p-1">
              <ChatWindow 
                dialogId={activeDialog} 
                selectedPdf={selectedPdf} 
                setSelectedPdf={setSelectedPdf}
                onCitationsUpdate={setCitations}
              />
            </div>
            
            <CitationsPanel citations={citations} />
          </div>
        </div>
      </main>
    </div>
  );
}




