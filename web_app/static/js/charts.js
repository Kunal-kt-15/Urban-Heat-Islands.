/**
 * Interactive Chart.js and Canvas visualizations for UHI Platform.
 */

const Charts = {
  landcoverDonut: null,
  scenarioChart: null,
  unetHistoryChart: null,
  regHistoryChart: null,
  scatterChart: null,

  // 1. Land-Cover Donut Chart
  renderLandcoverDonut(canvasId, pcts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (this.landcoverDonut) {
      this.landcoverDonut.destroy();
    }

    const data = [
      pcts.impervious_pct || 0,
      pcts.vegetation_pct || 0,
      pcts.bare_soil_pct || 0,
      pcts.water_pct || 0
    ];

    this.landcoverDonut = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Impervious', 'Vegetation', 'Bare Soil', 'Water'],
        datasets: [{
          data: data,
          backgroundColor: ['#e63946', '#10b981', '#e9c46a', '#38bdf8'],
          borderColor: '#0d121d',
          borderWidth: 3,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '68%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%`
            }
          }
        }
      }
    });
  },

  // 2. Mitigation Scenarios Multi-Bar Chart
  renderScenarioChart(canvasId, scenarios) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (this.scenarioChart) {
      this.scenarioChart.destroy();
    }

    const labels = scenarios.map(s => s.Scenario);
    const cooling = scenarios.map(s => s['Predicted Cooling (°C)']);
    const impPct = scenarios.map(s => s['Modified Impervious %']);
    const vegPct = scenarios.map(s => s['Modified Vegetation %']);

    this.scenarioChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Predicted Cooling (°C)',
            data: cooling,
            backgroundColor: '#10b981',
            borderRadius: 6,
            yAxisID: 'y'
          },
          {
            type: 'line',
            label: 'Impervious %',
            data: impPct,
            borderColor: '#e63946',
            borderWidth: 2,
            pointBackgroundColor: '#e63946',
            yAxisID: 'y1'
          },
          {
            type: 'line',
            label: 'Vegetation %',
            data: vegPct,
            borderColor: '#38bdf8',
            borderWidth: 2,
            pointBackgroundColor: '#38bdf8',
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Cooling Delta (°C)', color: '#10b981' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Surface Fraction (%)', color: '#94a3b8' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: { labels: { color: '#cbd5e1' } }
        }
      }
    });
  },

  // 3. Training Convergence Curves
  renderTrainingCurves(unetCanvasId, regCanvasId, unetHist, regHist) {
    const ctxU = document.getElementById(unetCanvasId);
    if (ctxU && unetHist && unetHist.train_loss) {
      if (this.unetHistoryChart) this.unetHistoryChart.destroy();
      const epochs = Array.from({ length: unetHist.train_loss.length }, (_, i) => i + 1);

      this.unetHistoryChart = new Chart(ctxU, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: unetHist.train_loss,
              borderColor: '#f97316',
              borderWidth: 2,
              tension: 0.2
            },
            {
              label: 'Val Loss',
              data: unetHist.val_loss,
              borderColor: '#fbbf24',
              borderWidth: 2,
              borderDash: [4, 4],
              tension: 0.2
            },
            {
              label: 'Val mIoU',
              data: unetHist.val_miou,
              borderColor: '#10b981',
              borderWidth: 2,
              tension: 0.2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#cbd5e1' } } }
        }
      });
    }

    const ctxR = document.getElementById(regCanvasId);
    if (ctxR && regHist && regHist.train_loss) {
      if (this.regHistoryChart) this.regHistoryChart.destroy();
      const epochsR = Array.from({ length: regHist.train_loss.length }, (_, i) => i + 1);

      this.regHistoryChart = new Chart(ctxR, {
        type: 'line',
        data: {
          labels: epochsR,
          datasets: [
            {
              label: 'Train Loss (MSE)',
              data: regHist.train_loss,
              borderColor: '#38bdf8',
              borderWidth: 2,
              tension: 0.2
            },
            {
              label: 'Val RMSE (°C)',
              data: regHist.val_rmse,
              borderColor: '#f43f5e',
              borderWidth: 2,
              tension: 0.2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#cbd5e1' } } }
        }
      });
    }
  },

  // 4. Render Confusion Matrix HTML Table
  renderConfusionMatrix(containerId, cm) {
    const container = document.getElementById(containerId);
    if (!container || !cm) return;

    const classes = ['Impervious', 'Vegetation', 'Bare Soil', 'Water'];
    let html = `<table class="data-table"><thead><tr><th>Actual \\ Predicted</th>`;
    classes.forEach(c => html += `<th>${c}</th>`);
    html += `</tr></thead><tbody>`;

    for (let r = 0; r < cm.length; r++) {
      html += `<tr><td><b>${classes[r]}</b></td>`;
      const rowSum = cm[r].reduce((a, b) => a + b, 0) || 1;
      for (let c = 0; c < cm[r].length; c++) {
        const val = cm[r][c];
        const pct = ((val / rowSum) * 100).toFixed(1);
        const isDiag = (r === c);
        const bg = isDiag ? 'rgba(16, 185, 129, 0.2)' : 'transparent';
        const color = isDiag ? '#34d399' : '#94a3b8';
        html += `<td style="background:${bg}; color:${color}; font-weight:${isDiag ? '700' : '400'}">${pct}% <span style="font-size:0.7rem; color:#64748b;">(${val})</span></td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
    container.innerHTML = html;
  }
};
