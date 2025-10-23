"use client";
import Link from "next/link";

export default function Home() {
  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">LexiAI</h1>
      <p className="mb-4">ברוך הבא ל־LexiAI – עוזר משפטי חכם</p>
      <div className="space-x-4">
        <Link href="/chat" className="text-blue-600 underline">
          מעבר לצ'אט
        </Link>
        <Link href="/predict" className="text-blue-600 underline">
          חיזוי תיקים
        </Link>
      </div>
    </main>
  );
}
