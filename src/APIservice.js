export class ApiService {
  static async analyzeECGSignal(rawSignalArray) {
    if (!Array.isArray(rawSignalArray) || rawSignalArray.length !== 4800 || rawSignalArray.some((row) => !Array.isArray(row) || row.length !== 12 || !row.every(Number.isFinite))) {
      throw new Error('Invalid ECG payload: expected exactly 4800 finite samples with 12 leads each.');
    }

    let response;
    try {
      response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal: rawSignalArray })
      });
    } catch (error) {
      throw new Error(`Local inference proxy is unavailable: ${error.message}`);
    }

    const payload = await response.json().catch(() => null);
    if (response.status === 405) {
      throw new Error('Inference proxy route is unavailable. Live Server cannot handle inference requests; run "python3 server.py" and open http://127.0.0.1:8765.');
    }
    if (!response.ok) throw new Error(payload?.error || `Inference proxy failed (${response.status}).`);
    if (!payload || typeof payload !== 'object') throw new Error('Inference proxy returned an invalid response.');
    return payload;
  }
}
