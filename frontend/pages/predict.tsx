"use client";
import { useState } from "react";

export default function PredictPage() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");

  const submit = async () => {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: input }),
    });
    const data = await res.json();
    setResult(JSON.stringify(data));
  };

  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">חיזוי תיקים</h1>
      <textarea
        className="border p-2 w-full h-24"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="הזן פרטי תיק לתחזית"
      />
      <button
        onClick={submit}
        className="bg-blue-600 text-white px-4 py-2 mt-2 rounded"
      >
        חשב חיזוי
      </button>
      {result && (
        <pre className="mt-4 p-4 border rounded bg-gray-50">{result}</pre>
      )}
    </main>
  );
}
