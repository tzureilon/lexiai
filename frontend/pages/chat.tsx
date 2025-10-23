"use client";
import { useState } from "react";

export default function ChatPage() {
  const [msg, setMsg] = useState("");
  const [chat, setChat] = useState<string[]>([]);

  const send = async () => {
    const res = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: "demo", message: msg }),
    });
    const data = await res.json();
    setChat([...chat, "👤 " + msg, "🤖 " + data.response]);
    setMsg("");
  };

  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">LexiAI Chat</h1>
      <div className="space-y-2 border p-4 rounded h-80 overflow-y-auto bg-gray-50">
        {chat.map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
      <div className="flex mt-4">
        <input
          className="border p-2 flex-1"
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
        />
        <button onClick={send} className="bg-blue-600 text-white px-4 ml-2 rounded">
          Send
        </button>
      </div>
    </main>
  );
}
