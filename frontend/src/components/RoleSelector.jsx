import React, { useEffect, useState } from "react";

const ROLE_ICONS = {
  humorous_butler: "🎩",
  scholarly_assistant: "📚",
  storyteller: "📖",
};

export default function RoleSelector() {
  const [roles, setRoles] = useState([]);
  const [currentRole, setCurrentRole] = useState(null);
  const [switching, setSwitching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  useEffect(() => {
    async function loadRoles() {
      try {
        const [rolesRes, currentRes] = await Promise.all([
          fetch("/api/chat/roles"),
          fetch("/api/chat/role/current"),
        ]);
        if (rolesRes.ok) {
          const data = await rolesRes.json();
          setRoles(data.roles || []);
        }
        if (currentRes.ok) {
          const data = await currentRes.json();
          setCurrentRole(data);
        }
      } catch (e) {
        console.error("加载角色失败:", e);
        setRoles([
          { role_id: "humorous_butler", display_name: "幽默的男管家", description: "英式管家..." },
          { role_id: "scholarly_assistant", display_name: "严谨的学术助手", description: "学术研究..." },
          { role_id: "storyteller", display_name: "博学的说书人", description: "走南闯北..." },
        ]);
        setCurrentRole({
          role_id: "humorous_butler",
          profile: { display_name: "幽默的男管家" },
        });
      }
    }
    loadRoles();
  }, []);

  async function switchRole(roleId) {
    if (switching) return;
    setSwitching(true);
    setShowDropdown(false);
    try {
      const res = await fetch("/api/chat/role/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role_id: roleId }),
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentRole({
          role_id: data.role_id,
          profile: { display_name: data.current_role },
        });
      }
    } catch (e) {
      console.error("切换角色失败:", e);
    } finally {
      setSwitching(false);
    }
  }

  const currentDisplayName = currentRole?.profile?.display_name || "加载中...";
  const currentRoleId = currentRole?.role_id || "";
  const icon = ROLE_ICONS[currentRoleId] || "🤖";

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 text-white hover:bg-white/20 transition-all duration-300 text-sm"
      >
        <span className="text-lg">{icon}</span>
        <span>{currentDisplayName}</span>
        <svg className={`w-4 h-4 transition-transform duration-300 ${showDropdown ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {showDropdown && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setShowDropdown(false)} />
          <div className="absolute right-0 mt-2 w-56 bg-slate-800 border border-white/20 rounded-xl shadow-2xl z-20 overflow-hidden">
            <div className="p-2 text-xs text-gray-400 px-3 py-2 border-b border-white/10">
              选择角色
            </div>
            {roles.map((role) => (
              <button
                key={role.role_id}
                onClick={() => switchRole(role.role_id)}
                disabled={switching}
                className={`w-full text-left px-4 py-3 text-sm flex items-center gap-3 transition-all duration-200 ${
                  role.role_id === currentRoleId
                    ? "bg-indigo-600/30 text-white"
                    : "text-gray-300 hover:bg-white/10 hover:text-white"
                } ${switching ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <span className="text-lg">{ROLE_ICONS[role.role_id] || "🤖"}</span>
                <div>
                  <div className="font-medium">{role.display_name}</div>
                  <div className="text-xs text-gray-500">{role.description}</div>
                </div>
                {role.role_id === currentRoleId && (
                  <svg className="w-4 h-4 text-green-400 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
