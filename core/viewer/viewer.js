import * as THREE from 'three';
import { SplatRenderer } from './splat_renderer.js';

// --- CONFIGURATION & STATE ---
let scene, camera, renderer;
let worldGroup;
let splatRenderer = null;

let trajectoryLine = null;
let kfFrustumsGroup = null;
let currentCamFrustum = null;

let isPolling = true;
let pollTimeout = null;
let lastSplatCount = 0;

// FPS Counter
let fpsLastTime = performance.now();
let fpsFrameCount = 0;

// HTML Elements
const statSplatCount = document.getElementById('stat-splat-count');
const statKfCount = document.getElementById('stat-kf-count');
const statFps = document.getElementById('stat-fps');
const valRefresh = document.getElementById('val-refresh');

const checkTrajectory = document.getElementById('control-show-trajectory');
const checkAutoRefresh = document.getElementById('control-auto-refresh');
const sliderRefreshRate = document.getElementById('control-refresh-rate');
const btnReset = document.getElementById('btn-reset');

// --- 4X4 MATRIX MATH HELPERS (From splat/main.js) ---
function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        (2 * fx) / width, 0, 0, 0,
        0, -(2 * fy) / height, 0, 0,
        0, 0, zfar / (zfar - znear), 1,
        0, 0, -(zfar * znear) / (zfar - znear), 0
    ];
}

// --- CAMERA NAVIGATION STATE (From splat/main.js) ---
const defaultViewMatrix = [
    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
    0.03, 6.55, 1,
];
let viewMatrix = [...defaultViewMatrix];
let activeKeys = [];
let startX = 0, startY = 0, down = false, altX = 0, altY = 0;
let splatCanvas;

// --- INITIALIZATION ---
function init() {
    const container = document.getElementById('canvas-container');

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 1000.0);
    camera.matrixAutoUpdate = false; // We drive camera strictly via custom view matrix!

    // Setup Three.js Renderer with transparent background!
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setClearColor(0x000000, 0); // Transparent
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    worldGroup = new THREE.Group();
    // No manual X rotation anymore, everything remains in the raw SLAM coordinate system!
    scene.add(worldGroup);

    const gridHelper = new THREE.GridHelper(20, 20, 0x374151, 0x1f2937);
    worldGroup.add(gridHelper);

    kfFrustumsGroup = new THREE.Group();
    worldGroup.add(kfFrustumsGroup);

    // Initialize splat/main.js Renderer
    splatCanvas = document.getElementById('splat-canvas');
    splatRenderer = new SplatRenderer(splatCanvas);

    setupUI();
    setupInteraction();
    onWindowResize(); // Force initial resize setup

    window.addEventListener('resize', onWindowResize);

    animate();
    pollData();
}

// --- INTERACTION EVENT LISTENERS (From splat/main.js) ---
function setupInteraction() {
    window.addEventListener("keydown", (e) => {
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();
            const lineHeight = 10;
            const scale = e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? window.innerHeight : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / window.innerWidth,
                    (e.deltaY * scale) / window.innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / window.innerHeight,
                );
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / window.innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / window.innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    splatCanvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    splatCanvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    splatCanvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / window.innerWidth;
            let dy = (5 * (e.clientY - startY)) / window.innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / window.innerWidth,
                0,
                (10 * (e.clientY - startY)) / window.innerHeight,
            );
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });

    splatCanvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    splatCanvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );

    splatCanvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / window.innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / window.innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx = (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
                const dy = (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
                let inv = invert4(viewMatrix);
                inv = rotate4(inv, dtheta, 0, 0, 1);
                inv = translate4(inv, -dx / window.innerWidth, -dy / window.innerHeight, 0);
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );

    splatCanvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );
}

// --- UI SETUP & BINDINGS ---
function setupUI() {
    const btnTogglePanel = document.getElementById('btn-toggle-panel');
    const controlPanelWrapper = document.getElementById('control-panel-wrapper');
    if (btnTogglePanel && controlPanelWrapper) {
        btnTogglePanel.addEventListener('click', () => {
            controlPanelWrapper.classList.toggle('collapsed');
            btnTogglePanel.textContent = controlPanelWrapper.classList.contains('collapsed') ? '▶' : '◀';
        });
    }

    checkTrajectory.addEventListener('change', (e) => {
        const visible = e.target.checked;
        if (trajectoryLine) trajectoryLine.visible = visible;
        kfFrustumsGroup.visible = visible;
        if (currentCamFrustum) currentCamFrustum.visible = visible;
    });

    checkAutoRefresh.addEventListener('change', (e) => {
        isPolling = e.target.checked;
        if (isPolling) {
            pollData();
        } else if (pollTimeout) {
            clearTimeout(pollTimeout);
        }
    });

    sliderRefreshRate.addEventListener('input', (e) => {
        valRefresh.textContent = `${e.target.value}ms`;
    });

    btnReset.addEventListener('click', () => {
        viewMatrix = [...defaultViewMatrix];
    });
}

// --- RENDER LOOP ---
function animate() {
    requestAnimationFrame(animate);

    // --- Native keyboard/gamepad camera updates ---
    let inv = invert4(viewMatrix);
    let shiftKey =
        activeKeys.includes("Shift") ||
        activeKeys.includes("ShiftLeft") ||
        activeKeys.includes("ShiftRight");

    if (activeKeys.includes("ArrowUp")) {
        if (shiftKey) {
            inv = translate4(inv, 0, -0.03, 0);
        } else {
            inv = translate4(inv, 0, 0, 0.1);
        }
    }
    if (activeKeys.includes("ArrowDown")) {
        if (shiftKey) {
            inv = translate4(inv, 0, 0.03, 0);
        } else {
            inv = translate4(inv, 0, 0, -0.1);
        }
    }
    if (activeKeys.includes("ArrowLeft"))
        inv = translate4(inv, -0.03, 0, 0);
    if (activeKeys.includes("ArrowRight"))
        inv = translate4(inv, 0.03, 0, 0);
        
    if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
    if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
    if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
    if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
    if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
    if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

    if (["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))) {
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(
            inv,
            activeKeys.includes("KeyJ")
                ? -0.05
                : activeKeys.includes("KeyL")
                  ? 0.05
                  : 0,
            0,
            1,
            0,
        );
        inv = rotate4(
            inv,
            activeKeys.includes("KeyI")
                ? 0.05
                : activeKeys.includes("KeyK")
                  ? -0.05
                  : 0,
            1,
            0,
            0,
        );
        inv = translate4(inv, 0, 0, -d);
    }

    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (let gamepad of gamepads) {
        if (!gamepad) continue;
        const axisThreshold = 0.1;
        const moveSpeed = 0.06;
        if (Math.abs(gamepad.axes[0]) > axisThreshold) {
            inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
        }
        if (Math.abs(gamepad.axes[1]) > axisThreshold) {
            inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
        }
        if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
            inv = translate4(inv, 0, -moveSpeed * (gamepad.buttons[12].pressed - gamepad.buttons[13].pressed), 0);
        }
        if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
            inv = translate4(inv, -moveSpeed * (gamepad.buttons[14].pressed - gamepad.buttons[15].pressed), 0, 0);
        }
    }

    viewMatrix = invert4(inv);

    // Calculate dynamic focal length & projection matrix exactly matching splat/main.js
    const h = window.innerHeight;
    const w = window.innerWidth;
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const fy = (h / 2.0) / Math.tan(fovRad / 2.0);
    const fx = fy;

    const projectionMatrix = getProjectionMatrix(fx, fy, w, h);

    // Sync Three.js Camera matrices perfectly with viewMatrix and custom projectionMatrix!
    camera.projectionMatrix.fromArray(projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

    camera.matrixWorldInverse.fromArray(viewMatrix);
    camera.matrixWorld.copy(camera.matrixWorldInverse).invert();
    camera.updateMatrixWorld(true);

    fpsFrameCount++;
    const now = performance.now();
    if (now >= fpsLastTime + 1000) {
        const fps = (fpsFrameCount * 1000) / (now - fpsLastTime);
        statFps.textContent = fps.toFixed(1);
        fpsFrameCount = 0;
        fpsLastTime = now;
    }

    // Render Three.js Trajectory on top
    renderer.render(scene, camera);

    // Render Gaussian Splats behind using the EXACT mathematical transforms of splat/main.js
    if (splatRenderer) {
        const viewProj = multiply4(projectionMatrix, viewMatrix);
        splatRenderer.render(viewProj, viewMatrix);
    }
}

function onWindowResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    
    renderer.setSize(w, h);
    
    if (splatRenderer) {
        splatRenderer.resize(w, h);
        
        // Calculate focal length in pixels for splat/main.js
        const fovRad = THREE.MathUtils.degToRad(camera.fov);
        const fy = (h / 2.0) / Math.tan(fovRad / 2.0);
        const fx = fy; 
        
        const projectionMatrix = getProjectionMatrix(fx, fy, w, h);
        splatRenderer.setProjection(projectionMatrix, fx, fy);
    }
}

// --- FRUSTUM WIREFRAME GENERATOR ---
function createFrustumSegments(c2w, scale = 0.12, color = 0x00ff00) {
    const vertices = [
        new THREE.Vector3(0, 0, 0),                              
        new THREE.Vector3(-0.5 * scale, -0.375 * scale, -scale),   
        new THREE.Vector3(0.5 * scale, -0.375 * scale, -scale),    
        new THREE.Vector3(0.5 * scale, 0.375 * scale, -scale),     
        new THREE.Vector3(-0.5 * scale, 0.375 * scale, -scale)     
    ];

    const mat = new THREE.Matrix4().fromArray(c2w);
    vertices.forEach(v => v.applyMatrix4(mat));

    const indices = [
        0, 1,  0, 2,  0, 3,  0, 4,
        1, 2,  2, 3,  3, 4,  4, 1  
    ];

    const points = [];
    indices.forEach(idx => {
        points.push(vertices[idx]);
    });

    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
    return new THREE.LineSegments(geom, material);
}

// --- DATA FETCHING ---
async function pollData() {
    if (!isPolling) return;

    try {
        const resPoses = await fetch('/poses');
        if (resPoses.ok) {
            const posesData = await resPoses.json();
            updateTrajectory(posesData);
        }

        const resGaussians = await fetch('/gaussians');
        if (resGaussians.ok) {
            const arrayBuffer = await resGaussians.arrayBuffer();
            updateGaussians(arrayBuffer);
        }

        const resLiveImage = await fetch('/live_image');
        if (resLiveImage.ok) {
            const blob = await resLiveImage.blob();
            const pipImg = document.getElementById('pip-img');
            if (pipImg && blob.size > 0) {
                const oldUrl = pipImg.src;
                pipImg.src = URL.createObjectURL(blob);
                if (oldUrl && oldUrl.startsWith('blob:')) {
                    setTimeout(() => URL.revokeObjectURL(oldUrl), 100);
                }
            }
        }
    } catch (e) {
        console.error("Polling error:", e);
    }

    const rate = parseInt(sliderRefreshRate.value);
    pollTimeout = setTimeout(pollData, rate);
}

function updateTrajectory(data) {
    const keyframes = data.keyframes || [];
    const currentPose = data.current_pose;

    statKfCount.textContent = keyframes.length;

    if (keyframes.length > 1) {
        const points = [];
        keyframes.forEach(kfMatrix => {
            points.push(new THREE.Vector3(kfMatrix[12], kfMatrix[13], kfMatrix[14]));
        });

        if (currentPose) {
            points.push(new THREE.Vector3(currentPose[12], currentPose[13], currentPose[14]));
        }

        if (trajectoryLine) worldGroup.remove(trajectoryLine);
        
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({ 
            color: 0x00d2ff, 
            linewidth: 3, 
            transparent: true,
            opacity: 0.8
        });
        trajectoryLine = new THREE.Line(geom, mat);
        trajectoryLine.visible = checkTrajectory.checked;
        worldGroup.add(trajectoryLine);
    }

    kfFrustumsGroup.clear();
    keyframes.forEach((kfMatrix) => {
        const frustum = createFrustumSegments(kfMatrix, 0.08, 0xffd54f);
        kfFrustumsGroup.add(frustum);
    });
    kfFrustumsGroup.visible = checkTrajectory.checked;

    if (currentPose) {
        if (currentCamFrustum) worldGroup.remove(currentCamFrustum);
        currentCamFrustum = createFrustumSegments(currentPose, 0.12, 0x00e676);
        currentCamFrustum.visible = checkTrajectory.checked;
        worldGroup.add(currentCamFrustum);

        // Dynamically align the viewer camera with the current SLAM camera pose!
        // We only do this if the user is not actively navigating with mouse/keyboard.
        if (!down && activeKeys.length === 0 && lastSplatCount > 0) {
            // Replicate the SLAM camera pose directly into the visualizer camera view matrix
            // Note: currentPose is the Camera-to-World (c2w) matrix, viewMatrix is World-to-Camera (w2c).
            // So we simply invert the currentPose!
            viewMatrix = invert4(currentPose);
        }
    }
}

function updateGaussians(arrayBuffer) {
    const rowLength = 32; 
    const numGaussians = Math.floor(arrayBuffer.byteLength / rowLength);
    statSplatCount.textContent = numGaussians.toLocaleString();

    if (numGaussians === 0) return;

    if (splatRenderer) {
        splatRenderer.updateSplats(arrayBuffer, numGaussians);
    }

    if (lastSplatCount === 0 && numGaussians > 0) {
        // Read the first Gaussian coordinate from the Float32Array to focus our viewMatrix correctly!
        const floatView = new Float32Array(arrayBuffer, 0, 3);
        const x = floatView[0];
        const y = floatView[1];
        const z = floatView[2];
        
        // Translate view matrix so that we focus directly on the loaded scene (instead of far off-center)
        let inv = invert4(defaultViewMatrix);
        // Place camera at a reasonable offset from the first Gaussian center
        inv[12] = x;
        inv[13] = y + 1.0;
        inv[14] = z + 4.0;
        viewMatrix = invert4(inv);
    }

    lastSplatCount = numGaussians;
}

init();
