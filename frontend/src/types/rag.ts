export interface RetrievedPair {
  context: string;
  response: string;
  score: number;
}

export interface RougeScores {
  [key: string]: number;
}

export interface Metrics {
  bleu: number;
  rouge: RougeScores;
}

export interface Evaluation {
  rag_response?: string;
  gemini_response?: string;
  ground_truth?: string;
  rag_metrics?: Metrics;
  gemini_metrics?: Metrics;
  error?: string;
}

export interface RagResponse {
  query: string;
  retrieved: RetrievedPair[];
  prompt: string;
  generated: string;
  evaluation?: Evaluation;
  disclaimer?: string;
}


