export class ApiService {
  static BACKEND_URL = "http://localhost:8000/api/v1/predict-ecg";

  static async analyzeECGSignal(rawSignalArray) {
    try {
      const response = await fetch(this.BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: "PT-90412",
          lead: "Lead_II",
          raw_signals: rawSignalArray
        })
      });

      if (!response.ok) throw new Error("Backend connection failed");
      return await response.json();
    } catch (error) {
      console.warn("Backend not active yet. Running local simulation.", error);
      return null;
    }
  }
}