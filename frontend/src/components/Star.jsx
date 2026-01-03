import React, { useState, useEffect, useRef } from "react";

export default function Star({
  id,
  activeIds = [],
  color = "white",
  size = "w-1 h-1",
  top,
  left,
  delay = "0s",
  duration = "3s",
  opacity = "70",
  onAnimationComplete
}) {
  const [showPing, setShowPing] = useState(false);
  const timeoutRef = useRef(null);

  const isActive = activeIds.includes(id);

  useEffect(() => {
    if (isActive) {
      setShowPing(true);
      timeoutRef.current = setTimeout(() => {
        setShowPing(false);
        if (onAnimationComplete) {
          onAnimationComplete();
        }
      }, 1500);
    } else {
      setShowPing(false);
    }
  }, [isActive, onAnimationComplete, color, size]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const colorClasses = {
    pink: "bg-pink-400",
    cyan: "bg-cyan-400",
    yellow: "bg-yellow-400",
    purple: "bg-purple-400",
    green: "bg-green-400",
    orange: "bg-orange-400",
    rose: "bg-rose-400",
    sky: "bg-sky-400",
    amber: "bg-amber-400",
    fuchsia: "bg-fuchsia-400",
    white: "bg-white",
  };

  const colorClass = colorClasses[color] || `bg-${color}-400`;

  const sizeMapping = {
    "w-1 h-1": "w-4 h-4",
    "w-1.5 h-1.5": "w-6 h-6",
    "w-2 h-2": "w-8 h-8",
  };

  const pingSize = sizeMapping[size] || size.replace(/\d+(\.5)?/, "4");
  const starOpacity = Math.round((parseInt(opacity) || 70) / 100 * 255);

  return (
    <div
      className="absolute"
      style={{ top, left }}
    >
      <div
        className={`${size} ${colorClass} rounded-full`}
        style={{
          opacity: starOpacity / 255,
          animationDelay: delay,
          animationDuration: duration,
        }}
      />
      {showPing && (
        <div
          className={`absolute inset-0 ${pingSize} ${colorClass} rounded-full animate-ping`}
          style={{
            animationDuration: "0.8s",
            opacity: 0.8,
          }}
        />
      )}
    </div>
  );
}
