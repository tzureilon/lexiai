"use client";
import { useState } from "react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");

  const upload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    setResult(`קובץ ${data.filename} הועלה (${data.size} בתים)`);
  };

  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">העלאת מסמך</h1>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <button onClick={upload} className="bg-blue-600 text-white px-4 ml-2 rounded">
        העלה
      </button>
      {result && <p className="mt-4">{result}</p>}
    </main>
  );
}
