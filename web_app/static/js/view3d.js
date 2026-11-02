/**
 * 3D Urban Digital Twin & Thermal Relief View
 * Powered by Three.js and WebGL
 */

const View3D = {
  container: null,
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  raycaster: null,
  mouse: null,

  // Mesh objects
  terrainMesh: null,
  buildingsGroup: null,
  waterMesh: null,
  treesGroup: null,
  gridHelper: null,
  dirLight: null,
  ambientLight: null,

  // State
  data: null,
  currentMode: 'semantic', // 'semantic', 'satellite', 'thermal', 'greening'
  heightScale: 1.5,
  greeningPct: 0.25,
  isAutoRotate: false,
  isWireframe: false,
  isInitialized: false,

  init(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    if (this.isInitialized) {
      this.onResize();
      return;
    }

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);
    this.scene.fog = new THREE.FogExp2(0x07090e, 0.0035);

    // 2. Camera
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 550;
    this.camera = new THREE.PerspectiveCamera(45, width / height, 1, 2000);
    this.camera.position.set(120, 140, 160);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.06;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
      this.controls.minDistance = 30;
      this.controls.maxDistance = 450;
      this.controls.target.set(0, 10, 0);
    }

    // 5. Lighting
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(this.ambientLight);

    this.dirLight = new THREE.DirectionalLight(0xfff4e0, 1.4);
    this.dirLight.position.set(120, 180, 90);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.width = 1024;
    this.dirLight.shadow.mapSize.height = 1024;
    this.dirLight.shadow.camera.near = 10;
    this.dirLight.shadow.camera.far = 500;
    const d = 120;
    this.dirLight.shadow.camera.left = -d;
    this.dirLight.shadow.camera.right = d;
    this.dirLight.shadow.camera.top = d;
    this.dirLight.shadow.camera.bottom = -d;
    this.scene.add(this.dirLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x07090e, 0.4);
    this.scene.add(hemiLight);

    // 6. Ground Reference Grid
    this.gridHelper = new THREE.GridHelper(240, 24, 0x1e3a5f, 0x0f172a);
    this.gridHelper.position.y = -0.5;
    this.scene.add(this.gridHelper);

    // 7. Raycaster for hover inspections
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.container.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('resize', () => this.onResize());

    this.isInitialized = true;
    this.animate();
  },

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  },

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.controls) {
      this.controls.autoRotate = this.isAutoRotate;
      this.controls.autoRotateSpeed = 1.2;
      this.controls.update();
    }
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  },

  /**
   * Loads 3D dataset and builds the city model
   */
  async loadData(sampleId) {
    try {
      this.showLoading(true);
      const res = await API.get3DMeshData(sampleId);
      this.data = res;
      this.buildModel();
      this.updateHUDStats();
      this.showLoading(false);
    } catch (e) {
      console.error('Failed to load 3D data:', e);
      this.showLoading(false);
    }
  },

  buildModel() {
    if (!this.data) return;

    // Clean up existing meshes
    if (this.buildingsGroup) this.scene.remove(this.buildingsGroup);
    if (this.treesGroup) this.scene.remove(this.treesGroup);
    if (this.terrainMesh) this.scene.remove(this.terrainMesh);
    if (this.waterMesh) this.scene.remove(this.waterMesh);

    const N = this.data.grid_size; // 64
    const spacing = 2.2;
    const totalSize = N * spacing;
    const halfSize = totalSize / 2;

    this.buildingsGroup = new THREE.Group();
    this.treesGroup = new THREE.Group();

    // Textures
    const textureLoader = new THREE.TextureLoader();
    let satTexture = null;
    let maskTexture = null;

    if (this.data.texture_rgb_base64) {
      satTexture = textureLoader.load(this.data.texture_rgb_base64);
    }
    if (this.data.texture_mask_base64) {
      maskTexture = textureLoader.load(this.data.texture_mask_base64);
    }

    // 1. Create Base Terrain Plane
    const terrainGeo = new THREE.PlaneGeometry(totalSize, totalSize, N - 1, N - 1);
    terrainGeo.rotateX(-Math.PI / 2);

    const pos = terrainGeo.attributes.position;
    const colors = new Float32Array(pos.count * 3);

    // Temperature bounds for thermal mode
    let minT = 20.0, maxT = 45.0;

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const idx = i * N + j;
        const cls = this.data.classes[i][j];
        const h = this.data.heights[i][j];
        const temp = this.data.temperatures[i][j];

        // In thermal mode, terrain height represents heat anomaly
        let vertexY = 0;
        if (this.currentMode === 'thermal') {
          vertexY = (temp - this.data.baseline_lst_c) * 4.5 * this.heightScale;
        } else if (cls === 3) {
          vertexY = -1.5; // Water basin
        } else {
          vertexY = 0.0;
        }

        pos.setY(idx, vertexY);

        // Vertex Colors based on mode
        let c = new THREE.Color();
        if (this.currentMode === 'thermal') {
          c = this.getThermalColor(temp, minT, maxT);
        } else if (this.currentMode === 'semantic') {
          c = this.getClassColor(cls);
        } else {
          c = new THREE.Color(0x223344);
        }

        colors[idx * 3] = c.r;
        colors[idx * 3 + 1] = c.g;
        colors[idx * 3 + 2] = c.b;
      }
    }

    terrainGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    terrainGeo.computeVertexNormals();

    let terrainMat;
    if (this.currentMode === 'satellite' && satTexture) {
      terrainMat = new THREE.MeshStandardMaterial({
        map: satTexture,
        roughness: 0.8,
        metalness: 0.1,
        wireframe: this.isWireframe
      });
    } else {
      terrainMat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.75,
        metalness: 0.15,
        wireframe: this.isWireframe
      });
    }

    this.terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
    this.terrainMesh.receiveShadow = true;
    this.scene.add(this.terrainMesh);

    // 2. Extrude 3D Buildings & Vegetation (if not in pure thermal terrain mode)
    if (this.currentMode !== 'thermal') {
      const boxGeo = new THREE.BoxGeometry(spacing * 0.88, 1, spacing * 0.88);
      boxGeo.translate(0, 0.5, 0); // Origin at bottom

      const treeGeo = new THREE.ConeGeometry(spacing * 0.55, 1, 6);
      treeGeo.translate(0, 0.5, 0);

      // Determine greening conversion set if in greening mode
      const convertedIndices = new Set();
      if (this.currentMode === 'greening') {
        const imperviousCoords = [];
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            if (this.data.classes[i][j] === 0) imperviousCoords.push({ i, j, dist: this.data.heights[i][j] });
          }
        }
        // Convert top % based on greeningPct
        const numConvert = Math.floor(imperviousCoords.length * this.greeningPct);
        imperviousCoords.sort((a, b) => b.dist - a.dist);
        for (let k = 0; k < numConvert; k++) {
          convertedIndices.add(`${imperviousCoords[k].i}_${imperviousCoords[k].j}`);
        }
      }

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const cls = this.data.classes[i][j];
          let h = this.data.heights[i][j] * (this.heightScale * 0.7);
          const x = j * spacing - halfSize;
          const z = i * spacing - halfSize;
          const isConverted = convertedIndices.has(`${i}_${j}`);

          if (cls === 0) {
            // Impervious Surface / Building
            if (isConverted) {
              // Converted into 3D Green Rooftop / Park!
              const greenMat = new THREE.MeshStandardMaterial({
                color: 0x10b981,
                roughness: 0.5,
                metalness: 0.1
              });
              const bMesh = new THREE.Mesh(boxGeo, greenMat);
              bMesh.scale.set(1, Math.max(2.0, h * 0.4), 1);
              bMesh.position.set(x, 0, z);
              bMesh.castShadow = true;
              bMesh.receiveShadow = true;
              bMesh.userData = { i, j, cls: 1, type: 'Simulated Green Park/Roof', h: h * 0.4, temp: this.data.temperatures[i][j] - 2.8 };
              this.buildingsGroup.add(bMesh);

              // Add tree on top
              const treeMat = new THREE.MeshStandardMaterial({ color: 0x059669, roughness: 0.7 });
              const tMesh = new THREE.Mesh(treeGeo, treeMat);
              tMesh.scale.set(1, 4.0, 1);
              tMesh.position.set(x, h * 0.4, z);
              tMesh.castShadow = true;
              this.treesGroup.add(tMesh);
            } else {
              // Regular Building Block
              let bColor = 0xe63946; // Red semantic
              if (this.currentMode === 'satellite') {
                bColor = 0xc0c8d4; // Architecture grey in sat mode
              }
              const bMat = new THREE.MeshStandardMaterial({
                color: bColor,
                roughness: 0.45,
                metalness: 0.25,
                wireframe: this.isWireframe
              });
              const bMesh = new THREE.Mesh(boxGeo, bMat);
              bMesh.scale.set(1, h, 1);
              bMesh.position.set(x, 0, z);
              bMesh.castShadow = true;
              bMesh.receiveShadow = true;
              bMesh.userData = { i, j, cls: 0, type: 'Impervious Building Block', h: h, temp: this.data.temperatures[i][j] };
              this.buildingsGroup.add(bMesh);
            }
          } else if (cls === 1) {
            // Natural Tree Canopy
            const treeMat = new THREE.MeshStandardMaterial({
              color: 0x10b981,
              roughness: 0.65,
              metalness: 0.1,
              wireframe: this.isWireframe
            });
            const tMesh = new THREE.Mesh(treeGeo, treeMat);
            tMesh.scale.set(1, Math.max(3.5, h * 1.5), 1);
            tMesh.position.set(x, 0, z);
            tMesh.castShadow = true;
            tMesh.userData = { i, j, cls: 1, type: 'Vegetation Canopy / Forest', h: h, temp: this.data.temperatures[i][j] };
            this.treesGroup.add(tMesh);
          }
        }
      }

      this.scene.add(this.buildingsGroup);
      this.scene.add(this.treesGroup);
    }

    // 3. Reflective Water Plane for water bodies
    const waterGeo = new THREE.PlaneGeometry(totalSize, totalSize);
    waterGeo.rotateX(-Math.PI / 2);
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.1,
      metalness: 0.8,
      transparent: true,
      opacity: 0.85
    });
    this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
    this.waterMesh.position.y = -0.6;
    this.scene.add(this.waterMesh);
  },

  /**
   * Heatmap color gradient: Blue -> Cyan -> Green -> Yellow -> Red
   */
  getThermalColor(temp, minT, maxT) {
    const t = THREE.MathUtils.clamp((temp - minT) / (maxT - minT), 0, 1);
    const color = new THREE.Color();
    if (t < 0.25) {
      color.lerpColors(new THREE.Color(0x3b82f6), new THREE.Color(0x06b6d4), t / 0.25);
    } else if (t < 0.5) {
      color.lerpColors(new THREE.Color(0x06b6d4), new THREE.Color(0x10b981), (t - 0.25) / 0.25);
    } else if (t < 0.75) {
      color.lerpColors(new THREE.Color(0x10b981), new THREE.Color(0xf59e0b), (t - 0.5) / 0.25);
    } else {
      color.lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), (t - 0.75) / 0.25);
    }
    return color;
  },

  getClassColor(cls) {
    switch (cls) {
      case 0: return new THREE.Color(0xe63946); // Red Impervious
      case 1: return new THREE.Color(0x10b981); // Green Vegetation
      case 2: return new THREE.Color(0xe9c46a); // Amber Soil
      case 3: return new THREE.Color(0x38bdf8); // Cyan Water
      default: return new THREE.Color(0x64748b);
    }
  },

  setMode(mode) {
    this.currentMode = mode;
    this.buildModel();
  },

  setHeightScale(scale) {
    this.heightScale = scale;
    this.buildModel();
  },

  setGreeningPct(pct) {
    this.greeningPct = pct;
    if (this.currentMode === 'greening') {
      this.buildModel();
    }
  },

  toggleAutoRotate(enable) {
    this.isAutoRotate = enable;
  },

  toggleWireframe(enable) {
    this.isWireframe = enable;
    this.buildModel();
  },

  resetCamera() {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(120, 140, 160);
    this.controls.target.set(0, 10, 0);
    this.controls.update();
  },

  setTopDownView() {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(0, 220, 0.1);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  },

  setIsometricView() {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(150, 120, 150);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  },

  onMouseMove(e) {
    if (!this.container || !this.camera || !this.raycaster) return;
    const rect = this.container.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const targets = [];
    if (this.buildingsGroup) targets.push(...this.buildingsGroup.children);
    if (this.treesGroup) targets.push(...this.treesGroup.children);
    if (this.terrainMesh) targets.push(this.terrainMesh);

    const intersects = this.raycaster.intersectObjects(targets, false);
    const tooltip = document.getElementById('twin-3d-tooltip');

    if (intersects.length > 0 && tooltip) {
      const hit = intersects[0];
      const ud = hit.object.userData;
      if (ud && ud.type) {
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top + 15}px`;

        const badgeClass = ud.cls === 0 ? 'badge-urban' : (ud.cls === 1 ? 'badge-veg' : 'badge-soil');
        tooltip.innerHTML = `
          <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.25rem;">
            <span class="${badgeClass}">●</span> ${ud.type}
          </div>
          <div style="font-size:0.75rem; color:#94a3b8; line-height:1.4;">
            <div>Grid Coord: [${ud.i}, ${ud.j}]</div>
            <div>Structure Height: <b>${ud.h.toFixed(1)} m</b></div>
            <div>Surface Temp: <b style="color:${ud.temp > 35 ? '#f43f5e' : '#10b981'};">${ud.temp.toFixed(1)} °C</b></div>
          </div>
        `;
      } else {
        tooltip.style.display = 'none';
      }
    } else if (tooltip) {
      tooltip.style.display = 'none';
    }
  },

  updateHUDStats() {
    if (!this.data) return;
    const elBuildings = document.getElementById('hud-total-buildings');
    const elVeg = document.getElementById('hud-total-veg');
    const elPeak = document.getElementById('hud-peak-temp');
    const elBase = document.getElementById('hud-base-elev');

    if (elBuildings) elBuildings.textContent = this.data.impervious_count;
    if (elVeg) elVeg.textContent = this.data.vegetation_count;
    if (elBase) elBase.textContent = `${this.data.base_elevation_m} m`;

    if (elPeak) {
      let maxT = -999;
      for (let r of this.data.temperatures) {
        for (let v of r) if (v > maxT) maxT = v;
      }
      elPeak.textContent = `${maxT.toFixed(1)} °C`;
    }
  },

  showLoading(show) {
    const loader = document.getElementById('twin-3d-loader');
    if (loader) loader.style.display = show ? 'flex' : 'none';
  }
};

window.View3D = View3D;
