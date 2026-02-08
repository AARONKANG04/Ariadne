const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export type ForYouPaper = { mag_id: string; title: string };
export type ForYouResponse = { papers: ForYouPaper[]; count: number };

export type PaperInfo = { mag_id: string; title: string | null; doi_url: string | null; abstract: string | null };

export async function fetchForYou(n: number = 50): Promise<ForYouResponse> {
  const res = await fetch(`${API_BASE}/api/papers/for-you?n=${n}`);
  if (!res.ok) throw new Error('Failed to fetch For You papers');
  return res.json();
}

export async function fetchPaperInfo(magId: string): Promise<PaperInfo> {
  const params = new URLSearchParams({ mag_id: magId });
  const res = await fetch(`${API_BASE}/api/papers/paper-info?${params}`);
  if (!res.ok) {
    if (res.status === 404) throw new Error('Paper not found');
    throw new Error('Failed to fetch paper info');
  }
  return res.json();
}
