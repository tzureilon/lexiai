import { useState } from 'react';

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const uploadData = await uploadResponse.json();

      const analyzeResponse = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: uploadData.extracted_text }),
      });
      const analyzeData = await analyzeResponse.json();
      setAnalysis(JSON.parse(analyzeData.analysis));
    } catch (error) {
      console.error('Error analyzing document:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Analyze Document</h1>
      <div className="flex items-center space-x-4 mb-4">
        <input type="file" onChange={handleFileChange} />
        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {analysis && (
        <div className="bg-gray-100 p-4 rounded">
          <h2 className="text-xl font-bold mb-2">Analysis Results</h2>
          <p><strong>Document Type:</strong> {analysis.document_type}</p>
          <p><strong>Parties:</strong> {analysis.parties}</p>
          <p><strong>Dates:</strong> {analysis.dates}</p>
        </div>
      )}
    </div>
  );
}
