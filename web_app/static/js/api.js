/**
 * REST API Client for Urban Heat Island (UHI) AI Mitigation Platform.
 */

const API = {
  baseUrl: '',

  async getHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getSamples() {
    const res = await fetch(`${this.baseUrl}/api/samples`);
    if (!res.ok) throw new Error('Failed to load sample areas');
    return res.json();
  },

  async getSampleData(sampleId) {
    const res = await fetch(`${this.baseUrl}/api/sample/${sampleId}`);
    if (!res.ok) throw new Error(`Failed to load sample ${sampleId}`);
    return res.json();
  },

  async runSegmentation(sampleId, file = null) {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else if (sampleId) {
      formData.append('sample_id', sampleId);
    }
    const res = await fetch(`${this.baseUrl}/api/segment`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Segmentation inference failed');
    return res.json();
  },

  async predictLST(features) {
    const res = await fetch(`${this.baseUrl}/api/predict-lst`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });
    if (!res.ok) throw new Error('LST prediction failed');
    return res.json();
  },

  async runSimulation(payload) {
    const res = await fetch(`${this.baseUrl}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Mitigation simulation failed');
    return res.json();
  },

  async getMultiScenarios(payload) {
    const res = await fetch(`${this.baseUrl}/api/multi-scenarios`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Failed to fetch multi-scenarios');
    return res.json();
  },

  async getMetrics() {
    const res = await fetch(`${this.baseUrl}/api/metrics`);
    if (!res.ok) throw new Error('Failed to fetch model metrics');
    return res.json();
  },

  async getZonalGrid(cellSize = 0.01) {
    const res = await fetch(`${this.baseUrl}/api/zonal-grid?cell_size=${cellSize}`);
    if (!res.ok) throw new Error('Failed to compute zonal grid');
    return res.json();
  },

  async get3DMeshData(sampleId) {
    const res = await fetch(`${this.baseUrl}/api/3d-data/${sampleId}`);
    if (!res.ok) throw new Error(`Failed to load 3D mesh data for ${sampleId}`);
    return res.json();
  }
};
