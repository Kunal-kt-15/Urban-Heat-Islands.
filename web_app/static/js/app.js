/**
 * Main Controller for Urban Heat Island (UHI) AI Mitigation Platform.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Global State
  const state = {
    currentTab: 'overview',
    currentSampleId: 'sample_area_1',
    activeSubtab: 'mask',
    samples: [],
    currentSampleData: null,
    metrics: null,
  };

  // 1. Navigation Tab Switching
  const navBtns = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabTarget = btn.getAttribute('data-tab');
      navBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(`tab-${tabTarget}`);
      if (targetPane) targetPane.classList.add('active');
      state.currentTab = tabTarget;

      // Trigger redraws if necessary
      if (tabTarget === 'analytics' && state.metrics) {
        renderAnalyticsTab();
      }
      if (tabTarget === 'twin3d') {
        init3DTab();
      }
    });
  });

  // 2. Hardware Telemetry & Health Check
  async function initTelemetry() {
    try {
      const health = await API.getHealth();
      const hw = health.hardware || {};
      const statusPill = document.getElementById('telemetry-status');
      if (hw.cuda_available) {
        statusPill.innerHTML = `<span class="status-dot"></span> NVIDIA ${hw.current_device} (CUDA) | Active`;
      } else {
        statusPill.innerHTML = `<span class="status-dot" style="background:#38bdf8;"></span> CPU Accelerated Inference`;
      }
    } catch (e) {
      console.warn('Telemetry init error:', e);
    }
  }

  // 3. Load Sample Areas
  async function loadSamplesList() {
    try {
      const res = await API.getSamples();
      state.samples = res.samples || [];
      const selectBoxes = [
        document.getElementById('sample-select-vision'),
        document.getElementById('sample-select-sim'),
        document.getElementById('sample-select-3d')
      ];

      selectBoxes.forEach(sb => {
        if (!sb) return;
        sb.innerHTML = '';
        state.samples.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.id;
          opt.textContent = `${s.name} (Urban: ${s.impervious_pct}% | Veg: ${s.vegetation_pct}%)`;
          sb.appendChild(opt);
        });
      });

      // Load initial sample
      await loadSelectedSample(state.currentSampleId);
    } catch (e) {
      console.error('Failed to load samples:', e);
    }
  }

  async function loadSelectedSample(sampleId) {
    try {
      state.currentSampleId = sampleId;
      const data = await API.getSampleData(sampleId);
      state.currentSampleData = data;

      // Update Vision Tab Images
      updateVisionImages(data);
      updateVisionMetrics(data.landcover_percentages);

      // Update Simulator Tab Initial State
      updateSimulatorOriginal(data);

      // Pre-fill Thermal Predictor sliders with sample metadata
      updatePredictorInputs(data);
    } catch (e) {
      console.error(`Error loading sample ${sampleId}:`, e);
    }
  }

  // 4. Satellite Vision Tab Logic
  function updateVisionImages(data) {
    const imgEl = document.getElementById('vision-display-img');
    if (!imgEl) return;

    if (state.activeSubtab === 'mask') {
      imgEl.src = data.mask_base64;
    } else if (state.activeSubtab === 'overlay') {
      imgEl.src = data.overlay_base64;
    } else {
      imgEl.src = data.image_base64;
    }
  }

  function updateVisionMetrics(pcts) {
    document.getElementById('pct-urban').textContent = `${pcts.impervious_pct}%`;
    document.getElementById('pct-veg').textContent = `${pcts.vegetation_pct}%`;
    document.getElementById('pct-soil').textContent = `${pcts.bare_soil_pct}%`;
    document.getElementById('pct-water').textContent = `${pcts.water_pct}%`;

    Charts.renderLandcoverDonut('vision-donut-chart', pcts);
  }

  // Vision Sub-tabs (Mask, Overlay, Optical)
  const subtabBtns = document.querySelectorAll('.subtab-btn');
  subtabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      subtabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeSubtab = btn.getAttribute('data-view');
      if (state.currentSampleData) {
        updateVisionImages(state.currentSampleData);
      }
    });
  });

  const sampleSelectVision = document.getElementById('sample-select-vision');
  if (sampleSelectVision) {
    sampleSelectVision.addEventListener('change', (e) => {
      loadSelectedSample(e.target.value);
    });
  }

  // File Upload Segment Button
  const uploadInput = document.getElementById('file-upload-input');
  const runSegBtn = document.getElementById('btn-run-segmentation');
  if (runSegBtn) {
    runSegBtn.addEventListener('click', async () => {
      const file = uploadInput.files[0];
      runSegBtn.disabled = true;
      runSegBtn.innerHTML = `⏳ Running U-Net Inference...`;

      try {
        const result = await API.runSegmentation(state.currentSampleId, file);
        state.currentSampleData = {
          ...state.currentSampleData,
          ...result
        };
        updateVisionImages(result);
        updateVisionMetrics(result.landcover_percentages);
      } catch (err) {
        alert('Segmentation error: ' + err.message);
      } finally {
        runSegBtn.disabled = false;
        runSegBtn.innerHTML = `🚀 Run U-Net Segmentation`;
      }
    });
  }

  // 5. Thermal Predictor Tab Logic
  function updatePredictorInputs(data) {
    const pcts = data.landcover_percentages || {};
    document.getElementById('slider-elev').value = data.elevation_m || 215;
    document.getElementById('val-elev').textContent = `${data.elevation_m || 215} m`;

    document.getElementById('slider-hum').value = data.humidity_pct || 50;
    document.getElementById('val-hum').textContent = `${data.humidity_pct || 50}%`;

    document.getElementById('slider-sol').value = data.solar_irradiance_wm2 || 700;
    document.getElementById('val-sol').textContent = `${data.solar_irradiance_wm2 || 700} W/m²`;

    document.getElementById('input-urban').value = pcts.impervious_pct || 50;
    document.getElementById('input-veg').value = pcts.vegetation_pct || 25;
    document.getElementById('input-soil').value = pcts.bare_soil_pct || 15;
    document.getElementById('input-water').value = pcts.water_pct || 10;

    triggerThermalPrediction();
  }

  // Live slider event listeners
  const predSliders = ['elev', 'hum', 'sol'];
  predSliders.forEach(s => {
    const el = document.getElementById(`slider-${s}`);
    const valEl = document.getElementById(`val-${s}`);
    if (el) {
      el.addEventListener('input', () => {
        const unit = s === 'elev' ? 'm' : s === 'hum' ? '%' : 'W/m²';
        valEl.textContent = `${el.value} ${unit}`;
        triggerThermalPrediction();
      });
    }
  });

  const predInputs = ['input-urban', 'input-veg', 'input-soil', 'input-water'];
  predInputs.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => triggerThermalPrediction());
    }
  });

  async function triggerThermalPrediction() {
    const payload = {
      elevation_m: parseFloat(document.getElementById('slider-elev').value),
      humidity_pct: parseFloat(document.getElementById('slider-hum').value),
      solar_irradiance_wm2: parseFloat(document.getElementById('slider-sol').value),
      impervious_pct: parseFloat(document.getElementById('input-urban').value),
      vegetation_pct: parseFloat(document.getElementById('input-veg').value),
      bare_soil_pct: parseFloat(document.getElementById('input-soil').value),
      water_pct: parseFloat(document.getElementById('input-water').value)
    };

    try {
      const res = await API.predictLST(payload);
      const val = res.predicted_lst_anomaly;
      const displayVal = document.getElementById('pred-lst-val');
      const badge = document.getElementById('pred-lst-badge');

      displayVal.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)} °C`;
      badge.textContent = res.heat_risk_level;
      badge.style.backgroundColor = `${res.risk_color}22`;
      badge.style.color = res.risk_color;
      badge.style.border = `1px solid ${res.risk_color}55`;
    } catch (e) {
      console.warn('Prediction error:', e);
    }
  }

  // 6. Mitigation Simulator Tab Logic
  function updateSimulatorOriginal(data) {
    const origImg = document.getElementById('sim-orig-img');
    if (origImg) origImg.src = data.mask_base64;
    triggerSimulation();
  }

  const sampleSelectSim = document.getElementById('sample-select-sim');
  if (sampleSelectSim) {
    sampleSelectSim.addEventListener('change', (e) => {
      loadSelectedSample(e.target.value);
    });
  }

  const simRatioSlider = document.getElementById('sim-ratio-slider');
  const simRatioVal = document.getElementById('sim-ratio-val');
  const simContiguous = document.getElementById('sim-contiguous-toggle');

  if (simRatioSlider) {
    simRatioSlider.addEventListener('input', () => {
      simRatioVal.textContent = `${simRatioSlider.value}%`;
      triggerSimulation();
    });
  }

  if (simContiguous) {
    simContiguous.addEventListener('change', () => triggerSimulation());
  }

  async function triggerSimulation() {
    if (!state.currentSampleData) return;

    const convPct = parseFloat(simRatioSlider.value) / 100.0;
    const contiguous = simContiguous.checked;

    const payload = {
      sample_id: state.currentSampleId,
      conversion_pct: convPct,
      contiguous: contiguous,
      elevation_m: state.currentSampleData.elevation_m,
      humidity_pct: state.currentSampleData.humidity_pct,
      solar_irradiance_wm2: state.currentSampleData.solar_irradiance_wm2
    };

    try {
      const res = await API.runSimulation(payload);
      // Update Images
      document.getElementById('sim-orig-img').src = res.original_mask_base64;
      document.getElementById('sim-mod-img').src = res.modified_mask_base64;

      // Update Key Metrics
      document.getElementById('sim-baseline-lst').textContent = `${res.original_lst_anomaly >= 0 ? '+' : ''}${res.original_lst_anomaly.toFixed(2)} °C`;
      document.getElementById('sim-modified-lst').textContent = `${res.modified_lst_anomaly >= 0 ? '+' : ''}${res.modified_lst_anomaly.toFixed(2)} °C`;
      document.getElementById('sim-cooling-delta').textContent = `-${res.predicted_temperature_reduction.toFixed(2)} °C`;

      document.getElementById('sim-px-converted').textContent = `${res.pixels_converted.toLocaleString()} px`;
      document.getElementById('sim-new-urban').textContent = `${res.modified_landcover_pcts.impervious_pct.toFixed(1)}%`;
      document.getElementById('sim-new-veg').textContent = `${res.modified_landcover_pcts.vegetation_pct.toFixed(1)}%`;

      // Trigger Multi-Scenario Analysis
      loadMultiScenarios();
    } catch (e) {
      console.error('Simulation error:', e);
    }
  }

  async function loadMultiScenarios() {
    try {
      const payload = {
        sample_id: state.currentSampleId,
        elevation_m: state.currentSampleData.elevation_m,
        humidity_pct: state.currentSampleData.humidity_pct,
        solar_irradiance_wm2: state.currentSampleData.solar_irradiance_wm2
      };
      const res = await API.getMultiScenarios(payload);
      const scenarios = res.scenarios || [];

      // Render Cards
      const container = document.getElementById('scenarios-cards-container');
      if (container) {
        container.innerHTML = '';
        scenarios.forEach(sc => {
          const card = document.createElement('div');
          card.className = 'scenario-card';
          card.innerHTML = `
            <div class="scenario-tag">Scenario ${sc.Scenario}</div>
            <div class="scenario-cooling">-${sc['Predicted Cooling (°C)'].toFixed(2)} °C</div>
            <div class="scenario-sub">Veg: ${sc['Modified Vegetation %'].toFixed(1)}% | Urban: ${sc['Modified Impervious %'].toFixed(1)}%</div>
          `;
          container.appendChild(card);
        });
      }

      // Render Multi-Scenario Chart
      Charts.renderScenarioChart('scenario-multi-chart', scenarios);
    } catch (e) {
      console.warn('Scenarios load error:', e);
    }
  }

  // 7. Performance & Analytics Tab
  async function loadMetricsData() {
    try {
      state.metrics = await API.getMetrics();
    } catch (e) {
      console.warn('Failed to load metrics:', e);
    }
  }

  function renderAnalyticsTab() {
    if (!state.metrics) return;
    const { unet_history, regression_history } = state.metrics;

    Charts.renderTrainingCurves(
      'chart-unet-history',
      'chart-reg-history',
      unet_history,
      regression_history
    );

    const testM = unet_history.test_metrics || {};
    if (testM.confusion_matrix) {
      Charts.renderConfusionMatrix('cm-container', testM.confusion_matrix);
    }
  }

  // 8. Geospatial Grid Tab
  const genGridBtn = document.getElementById('btn-gen-grid');
  const gridSizeSelect = document.getElementById('grid-size-select');

  if (genGridBtn) {
    genGridBtn.addEventListener('click', async () => {
      genGridBtn.disabled = true;
      genGridBtn.innerHTML = `⏳ Aggregating Grid Cells...`;
      try {
        const sz = parseFloat(gridSizeSelect.value);
        const res = await API.getZonalGrid(sz);
        renderZonalTable(res.records || []);
      } catch (e) {
        alert('Grid error: ' + e.message);
      } finally {
        genGridBtn.disabled = false;
        genGridBtn.innerHTML = `🗺️ Generate & Aggregate Zones`;
      }
    });
  }

  function renderZonalTable(records) {
    const tbody = document.getElementById('zonal-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    records.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>${r.cell_id}</b></td>
        <td>${r.impervious_pct}%</td>
        <td>${r.vegetation_pct}%</td>
        <td>${r.bare_soil_pct}%</td>
        <td>${r.water_pct}%</td>
        <td style="color:${r.mean_lst_anomaly > 3 ? '#f43f5e' : '#10b981'}; font-weight:700;">
          ${r.mean_lst_anomaly >= 0 ? '+' : ''}${r.mean_lst_anomaly.toFixed(2)} °C
        </td>
        <td>${r.elevation_m} m</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // 7. 3D Digital Twin Tab Controls
  let is3DTabInitialized = false;
  function init3DTab() {
    if (typeof View3D === 'undefined') return;
    if (!is3DTabInitialized) {
      View3D.init('twin-3d-viewport');
      bind3DControls();
      is3DTabInitialized = true;
    }
    const select3d = document.getElementById('sample-select-3d');
    const selectedId = select3d && select3d.value ? select3d.value : state.currentSampleId;
    View3D.loadData(selectedId || 'sample_area_1');
    setTimeout(() => View3D.onResize(), 120);
  }

  function bind3DControls() {
    // Mode buttons
    const modeBtns = document.querySelectorAll('.twin-mode-btn');
    const greeningWrapper = document.getElementById('wrapper-greening-slider');

    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.getAttribute('data-mode');
        View3D.setMode(mode);

        if (greeningWrapper) {
          greeningWrapper.style.display = (mode === 'greening') ? 'flex' : 'none';
        }
      });
    });

    // Sample selector
    const select3d = document.getElementById('sample-select-3d');
    if (select3d) {
      select3d.addEventListener('change', (e) => {
        View3D.loadData(e.target.value);
      });
    }

    // Height scale slider
    const sliderHeight = document.getElementById('slider-height-scale');
    const labelHeight = document.getElementById('label-height-scale');
    if (sliderHeight && labelHeight) {
      sliderHeight.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        labelHeight.textContent = `${val.toFixed(1)}x`;
        View3D.setHeightScale(val);
      });
    }

    // Greening slider
    const sliderGreen = document.getElementById('slider-greening-3d');
    const labelGreen = document.getElementById('label-greening-3d');
    if (sliderGreen && labelGreen) {
      sliderGreen.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        labelGreen.textContent = `${Math.round(val * 100)}%`;
        View3D.setGreeningPct(val);
      });
    }

    // Camera preset buttons
    const btnReset = document.getElementById('btn-cam-reset');
    if (btnReset) btnReset.addEventListener('click', () => View3D.resetCamera());

    const btnIso = document.getElementById('btn-cam-iso');
    if (btnIso) btnIso.addEventListener('click', () => View3D.setIsometricView());

    const btnTop = document.getElementById('btn-cam-top');
    if (btnTop) btnTop.addEventListener('click', () => View3D.setTopDownView());

    // Toggles
    const toggleAuto = document.getElementById('toggle-auto-rotate');
    if (toggleAuto) {
      toggleAuto.addEventListener('change', (e) => {
        View3D.toggleAutoRotate(e.target.checked);
      });
    }

    const toggleWire = document.getElementById('toggle-wireframe');
    if (toggleWire) {
      toggleWire.addEventListener('change', (e) => {
        View3D.toggleWireframe(e.target.checked);
      });
    }
  }

  // Initial Boot
  await initTelemetry();
  await loadSamplesList();
  await loadMetricsData();
});
