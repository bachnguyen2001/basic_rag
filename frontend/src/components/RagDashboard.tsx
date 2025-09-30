import { useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type { RagResponse, RetrievedPair } from "../types/rag";

function MetricRow({ label, value }: { label: string; value: number | string | undefined }) {
  return (
    <div className="metric-row">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "-"}</div>
    </div>
  );
}

export default function RagDashboard() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RagResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setData(null);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await api.queryRag(query, controller.signal);
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const retrieved: RetrievedPair[] = useMemo(() => data?.retrieved ?? [], [data]);

  return (
    <div className="container">
      <header className="header">
        <h1>RAG Evaluation Dashboard</h1>
        <div className="subtitle">Deep Learning Demo — Mental Health Counseling</div>
      </header>

      <form className="query-form" onSubmit={onSubmit}>
        <input
          className="query-input"
          placeholder="Nhập câu hỏi của bạn..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn" type="submit" disabled={loading || !query.trim()}>
          {loading ? "Đang xử lý..." : "Gửi"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {data && (
        <div className="grid">
          <section className="card">
            <h2>Kết quả sinh (RAG)</h2>
            <pre className="prewrap">{data.generated}</pre>
          </section>

          <section className="card">
            <h2>So sánh đánh giá</h2>
            <div className="metrics">
              <MetricRow label="BLEU (RAG)" value={data.evaluation?.rag_metrics?.bleu?.toFixed?.(4)} />
              <MetricRow label="BLEU (Gemini)" value={data.evaluation?.gemini_metrics?.bleu?.toFixed?.(4)} />
              <MetricRow label="ROUGE-1 (RAG)" value={data.evaluation?.rag_metrics?.rouge?.rouge1?.toFixed?.(4)} />
              <MetricRow label="ROUGE-L (RAG)" value={data.evaluation?.rag_metrics?.rouge?.rougeL?.toFixed?.(4)} />
              <MetricRow label="ROUGE-1 (Gemini)" value={data.evaluation?.gemini_metrics?.rouge?.rouge1?.toFixed?.(4)} />
              <MetricRow label="ROUGE-L (Gemini)" value={data.evaluation?.gemini_metrics?.rouge?.rougeL?.toFixed?.(4)} />
            </div>
            <details>
              <summary>Phản hồi Gemini (thuần)</summary>
              <pre className="prewrap">{data.evaluation?.gemini_response}</pre>
            </details>
            <details>
              <summary>Ground truth</summary>
              <pre className="prewrap">{data.evaluation?.ground_truth}</pre>
            </details>
          </section>

          <section className="card card-span">
            <h2>Ngữ cảnh truy hồi (top-k)</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Score</th>
                  <th>Context</th>
                  <th>Response</th>
                </tr>
              </thead>
              <tbody>
                {retrieved.map((r, idx) => (
                  <tr key={idx}>
                    <td>{idx + 1}</td>
                    <td>{r.score.toFixed(4)}</td>
                    <td className="mono">{r.context}</td>
                    <td className="mono">{r.response}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="card card-span">
            <h2>Prompt</h2>
            <details open>
              <summary>Hiển thị/Ẩn prompt</summary>
              <pre className="prewrap small">{data.prompt}</pre>
            </details>
          </section>

          {data.disclaimer && (
            <section className="disclaimer">{data.disclaimer}</section>
          )}
        </div>
      )}
    </div>
  );
}


