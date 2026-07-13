let inferenceSession;
let indexLabels;
let relationalDatabase;
let activeVideoStream = null;

// 1. Structural Application Boot Initialization Orchestrator
async function initializePipeline() {
    try {
        // Asynchronously fetch separate metadata components (Phase 3 Decoupling validation)
        const labelsResponse = await fetch('labels.json');
        indexLabels = await labelsResponse.json();

        const dataResponse = await fetch('static/data/flowers.json');
        relationalDatabase = await dataResponse.json();

        // Boot ONNX runtime engine
        inferenceSession = await ort.InferenceSession.create('./flower_model.onnx');
        updateStatus("System Status: AI Pipeline Active & Operational.");
    } catch (err) {
        updateStatus("System Status: Initialization Error occurred.");
        console.error(err);
    }
}

// 2. Setup Core DOM Element Selectors & Interface Anchors
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('imageSelector');
const previewImg = document.getElementById('preview');
const videoElement = document.getElementById('videoFeed');
const startCamBtn = document.getElementById('startCamBtn');
const captureBtn = document.getElementById('captureBtn');
const resetBtn = document.getElementById('resetBtn');
const themeToggle = document.getElementById('themeToggle');

// 3. User Interface Interlock Event Wireframes & Operational Hooks
themeToggle.addEventListener('click', () => {
    const activeTheme = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', activeTheme === 'dark' ? 'light' : 'dark');
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => processImageSource(e.target.files[0]));

// Drag and Drop Engine Hooks
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#d23669'; });
dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '');
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
    if (e.dataTransfer.files.length > 0) processImageSource(e.dataTransfer.files[0]);
});

// Camera Stream Handlers
startCamBtn.addEventListener('click', async () => {
    try {
        activeVideoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        videoElement.srcObject = activeVideoStream;
        videoElement.style.display = 'block';
        captureBtn.style.display = 'inline-block';
        startCamBtn.style.display = 'none';
        previewImg.style.display = 'none';
    } catch (err) {
        updateStatus("Camera access denied or unavailable.");
    }
});

captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    
    const dataUrl = canvas.toDataURL('image/jpeg');
    previewImg.src = dataUrl;
    previewImg.style.display = 'inline-block';
    
    // Halt active hardware camera stream tracking to save memory
    if (activeVideoStream) {
        activeVideoStream.getTracks().forEach(track => track.stop());
    }
    videoElement.style.display = 'none';
    captureBtn.style.display = 'none';
    startCamBtn.style.display = 'inline-block';

    const img = new Image();
    img.onload = () => runEdgeInference(img);
    img.src = dataUrl;
});

resetBtn.addEventListener('click', () => {
    previewImg.style.display = 'none';
    document.getElementById('flowerCard').style.display = 'none';
    document.getElementById('predictionsHub').style.display = 'none';
    resetBtn.style.display = 'none';
    fileInput.value = '';
    updateStatus("Workspace cleared. Awaiting input...");
});

function updateStatus(msg) { document.getElementById('status-box').innerText = msg; }

function processImageSource(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        previewImg.src = event.target.result;
        previewImg.style.display = 'inline-block';
        resetBtn.style.display = 'inline-block';
        
        const img = new Image();
        img.onload = () => runEdgeInference(img);
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

// 4. Mathematical Array Preprocessing Engine (Phase 4 Validation Rules)
// Scales layout buffers to 224x224 and standardizes normalizations to ImageNet guidelines
function preprocessFrame(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    // Compute central square bounding parameters to mirror PyTorch CenterCrop logic
    const shortSide = Math.min(img.width, img.height);
    const sourceX = (img.width - shortSide) / 2;
    const sourceY = (img.height - shortSide) / 2;
    
    ctx.drawImage(img, sourceX, sourceY, shortSide, shortSide, 0, 0, 224, 224);
    
    const pixelData = ctx.getImageData(0, 0, 224, 224).data;
    const floatArray = new Float32Array(3 * 224 * 224);

    // Exact standard global evaluation metrics formulas: (x - mean) / std Deviation
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
        let r = pixelData[i * 4] / 255.0;
        let g = pixelData[i * 4 + 1] / 255.0;
        let b = pixelData[i * 4 + 2] / 255.0;

        floatArray[i] = (r - mean[0]) / std[0];
        floatArray[i + 224 * 224] = (g - mean[1]) / std[1];
        floatArray[i + 2 * 224 * 224] = (b - mean[2]) / std[2];
    }
    return new ort.Tensor('float32', floatArray, [1, 3, 224, 224]);
}

// 5. Execution Inference Loop & Softmax Array Calculus Engine (Phase 2 Upgrade)
async function runEdgeInference(img) {
    updateStatus("Processing image matrix buffers...");
    const inputTensor = preprocessFrame(img);
    
    const outputMap = await inferenceSession.run({ input: inputTensor });
    const rawLogits = outputMap.output.data;

    // Convert raw logit scores into exponential distributions (Softmax Array Engine)
    const exponents = rawLogits.map(Math.exp);
    const sumExponents = exponents.reduce((a, b) => a + b, 0);
    const probabilities = exponents.map(val => val / sumExponents);

    // Sort confidence metrics mapping array structures to extract Top-3 outputs
    const rankedClasses = Array.from(probabilities)
        .map((prob, idx) => ({ name: indexLabels[idx], confidence: prob }))
        .sort((a, b) => b.confidence - a.confidence);

    renderConfidenceRankings(rankedClasses.slice(0, 3));
    renderTargetFlashcard(rankedClasses[0].name);
    appendHistoryLog(rankedClasses[0].name, rankedClasses[0].confidence);
    
    updateStatus("Analysis sequence complete.");
    resetBtn.style.display = 'inline-block';
}

// 6. Component Template Render Controls
function renderConfidenceRankings(topThree) {
    const container = document.getElementById('rankingsContainer');
    container.innerHTML = '';
    topThree.forEach(item => {
        const percentage = (item.confidence * 100).toFixed(2);
        const row = document.createElement('div');
        row.className = 'rank-row';
        row.innerHTML = `
            <div class="rank-meta"><span>${item.name}</span><span>${percentage}%</span></div>
            <div class="bar-track"><div class="bar-fill" style="width: ${percentage}%"></div></div>
        `;
        container.appendChild(row);
    });
    document.getElementById('predictionsHub').style.display = 'block';
}

function renderTargetFlashcard(targetName) {
    const plant = relationalDatabase[targetName];
    if (!plant) return;

    document.getElementById('cardTitle').innerText = targetName;
    document.getElementById('cardScientific').innerText = plant.scientific_name;
    document.getElementById('cardFamily').innerText = plant.family;
    document.getElementById('cardNative').innerText = plant.native;
    document.getElementById('cardSeason').innerText = plant.bloom_season;
    document.getElementById('cardSun').innerText = plant.sunlight;
    document.getElementById('cardWater').innerText = plant.water;
    document.getElementById('cardSoil').innerText = plant.soil;
    document.getElementById('cardToxic').innerText = plant.toxicity;
    document.getElementById('cardMeaning').innerText = plant.meaning;
    document.getElementById('cardFact').innerText = plant.fact;
    document.getElementById('cardCare').innerText = plant.care;
    
    const wikiLink = document.getElementById('cardWiki');
    wikiLink.href = plant.wikipedia;

    document.getElementById('flowerCard').style.display = 'block';
}

function appendHistoryLog(name, confidence) {
    const historyList = document.getElementById('historyList');
    const item = document.createElement('li');
    item.className = 'history-item';
    item.innerText = `🌸 ${name} detected (${(confidence * 100).toFixed(1)}%) at ${new Date().toLocaleTimeString()}`;
    historyList.insertBefore(item, historyList.firstChild);
}

// Kickstart App Engine
initializePipeline();