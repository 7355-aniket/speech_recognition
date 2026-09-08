/* Speech Recognition JavaScript Frontend Logic */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const btnRecord = document.getElementById('btnRecord');
    const micIcon = document.getElementById('micIcon');
    const micContainer = document.getElementById('micContainer');
    const recordStatus = document.getElementById('recordStatus');
    const waveformCanvas = document.getElementById('waveformCanvas');
    const spectrogramCanvas = document.getElementById('spectrogramCanvas');
    const transcriptionText = document.getElementById('transcriptionText');
    const sttConfidence = document.getElementById('sttConfidence');
    const tokenContainer = document.getElementById('tokenContainer');
    const assistantText = document.getElementById('assistantText');
    const assistantBadge = document.getElementById('assistantBadge');
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const sampleButtons = document.getElementById('sampleButtons');
    const btnTrainModel = document.getElementById('btnTrainModel');
    const audioDuration = document.getElementById('audioDuration');
    const audioRate = document.getElementById('audioRate');

    // Audio & Recording State
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyser = null;
    let animFrameId = null;

    // Web Speech API Native Fallback
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let webSpeechRec = null;
    if (SpeechRecognition) {
        webSpeechRec = new SpeechRecognition();
        webSpeechRec.continuous = false;
        webSpeechRec.interimResults = true;
        webSpeechRec.lang = 'en-US';

        webSpeechRec.onresult = (event) => {
            let interimText = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    const finalTranscript = event.results[i][0].transcript;
                    transcribeCompleted(finalTranscript, 0.96);
                } else {
                    interimText += event.results[i][0].transcript;
                    transcriptionText.innerText = `"${interimText}..."`;
                }
            }
        };

        webSpeechRec.onerror = (err) => {
            console.warn('WebSpeech API fallback error:', err);
        };
    }

    // Initialize Waveform Canvas Context
    const waveCtx = waveformCanvas.getContext('2d');
    const specCtx = spectrogramCanvas.getContext('2d');

    // Load Sample Audio Files
    loadSampleAudioFiles();

    // Event Listeners
    btnRecord.addEventListener('click', toggleRecording);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files[0]));

    // Drag and Drop Events
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-indigo-400'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-indigo-400'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-400');
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });

    // Preset Command Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const commandText = btn.innerText.replace(/"/g, '').trim();
            transcriptionText.innerText = `"${commandText}"`;
            executeVoiceCommand(commandText);
        });
    });

    // Train Model Button
    btnTrainModel.addEventListener('click', async () => {
        btnTrainModel.innerText = 'Training PyTorch Model...';
        btnTrainModel.disabled = true;
        try {
            const res = await fetch('/api/train_demo', { method: 'POST' });
            const data = await res.json();
            alert(`PyTorch Neural Model Training: ${data.message}`);
        } catch (e) {
            console.error(e);
        } finally {
            btnTrainModel.innerText = 'Train Neural Model';
            btnTrainModel.disabled = false;
        }
    });

    // Audio Recording Toggle
    async function toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    async function startRecording() {
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);

            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
            mediaRecorder.onstop = processRecordedAudio;

            mediaRecorder.start();
            if (webSpeechRec) webSpeechRec.start();

            isRecording = true;
            micIcon.className = 'fa-solid fa-square text-2xl';
            micContainer.classList.add('recording');
            recordStatus.innerText = 'Listening... Speak into microphone';
            recordStatus.className = 'mt-3 text-xs font-medium text-red-400 animate-pulse';

            drawWaveform();
        } catch (err) {
            console.error('Microphone access error:', err);
            recordStatus.innerText = 'Microphone access denied. Try uploading a sample WAV file.';
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            if (webSpeechRec) webSpeechRec.stop();

            isRecording = false;
            micIcon.className = 'fa-solid fa-microphone text-2xl';
            micContainer.classList.remove('recording');
            recordStatus.innerText = 'Processing speech audio...';
            recordStatus.className = 'mt-3 text-xs font-medium text-indigo-400';

            if (animFrameId) cancelAnimationFrame(animFrameId);
        }
    }

    // Process Recorded Audio Blob
    async function processRecordedAudio() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'mic_input.wav');

        sendAudioForTranscription(formData);
    }

    // Handle File Upload
    function handleFileUpload(file) {
        if (!file) return;
        recordStatus.innerText = `Processing ${file.name}...`;

        const formData = new FormData();
        formData.append('file', file);
        sendAudioForTranscription(formData);
    }

    // Send Audio File to Backend API
    async function sendAudioForTranscription(formData) {
        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.status === 'success' && data.result) {
                const res = data.result;
                transcribeCompleted(res.transcription, res.confidence, res.tokens, res.layer_activations);

                if (res.audio_features) {
                    renderSpectrogram(res.audio_features.spectrogram);
                    audioDuration.innerText = `Duration: ${res.audio_features.duration}s`;
                    audioRate.innerText = `Rate: ${res.audio_features.sample_rate} Hz`;
                }

                if (data.assistant_response) {
                    renderAssistantResponse(data.assistant_response);
                }
            } else {
                recordStatus.innerText = 'Speech recognition failed.';
            }
        } catch (e) {
            console.error('Transcription error:', e);
            recordStatus.innerText = 'Error connecting to backend STT server.';
        }
    }

    // Complete Transcription Callback
    function transcribeCompleted(text, confidence, tokens = [], layerActivations = null) {
        recordStatus.innerText = 'Click microphone to start speaking';
        recordStatus.className = 'mt-3 text-xs font-medium text-slate-400';

        if (text) {
            transcriptionText.innerText = `"${text}"`;
            const confPct = Math.round((confidence || 0.92) * 100);
            sttConfidence.innerText = `Confidence: ${confPct}%`;

            // Tokens display
            tokenContainer.innerHTML = '';
            const words = tokens.length ? tokens : text.split(' ').map(w => [w, 0.92]);
            words.forEach(([w, c]) => {
                const chip = document.createElement('span');
                chip.className = `token-chip ${c > 0.9 ? 'confidence-high' : 'confidence-med'}`;
                chip.innerHTML = `${w} <span class="text-[10px] opacity-70">${Math.round(c*100)}%</span>`;
                tokenContainer.appendChild(chip);
            });

            executeVoiceCommand(text);
        }
    }

    // Execute Assistant Voice Command
    async function executeVoiceCommand(text) {
        try {
            const res = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await res.json();
            if (data.status === 'success') {
                renderAssistantResponse(data.data);
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Render Assistant Voice Response
    function renderAssistantResponse(data) {
        if (!data) return;
        assistantText.innerText = data.response;
        assistantBadge.innerText = data.badge || data.intent;

        // Perform TTS Audio Speech Synthesis
        if ('speechSynthesis' in window && data.action === 'speak') {
            window.speechSynthesis.cancel(); // Stop prior speech
            const utterance = new SpeechSynthesisUtterance(data.response);
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }

        if (data.action === 'open_url' && data.url) {
            window.open(data.url, '_blank');
        }
    }

    // Load Sample Audio Preset Files
    async function loadSampleAudioFiles() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.samples && sampleButtons) {
                sampleButtons.innerHTML = '';
                data.samples.forEach((sample, idx) => {
                    const btn = document.createElement('button');
                    btn.className = 'text-xs px-3 py-1.5 bg-indigo-950/60 hover:bg-indigo-900 text-indigo-300 rounded-lg border border-indigo-800 transition flex items-center gap-1.5';
                    btn.innerHTML = `<i class="fa-solid fa-play text-[10px]"></i> Sample #${idx + 1}`;
                    btn.addEventListener('click', async () => {
                        recordStatus.innerText = `Loading ${sample.name}...`;
                        const audioRes = await fetch(sample.url);
                        const blob = await audioRes.getBlob();
                        const formData = new FormData();
                        formData.append('file', blob, sample.name);
                        sendAudioForTranscription(formData);
                    });
                    sampleButtons.appendChild(btn);
                });
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Render Waveform Canvas
    function drawWaveform() {
        if (!analyser) return;
        const bufferLength = analyser.fftSize;
        const dataArray = new Uint8Array(bufferLength);

        function render() {
            animFrameId = requestAnimationFrame(render);
            analyser.getByteTimeDomainData(dataArray);

            waveCtx.fillStyle = '#090d16';
            waveCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            waveCtx.lineWidth = 2;
            waveCtx.strokeStyle = '#6366f1';
            waveCtx.beginPath();

            const sliceWidth = waveformCanvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * waveformCanvas.height) / 2;

                if (i === 0) waveCtx.moveTo(x, y);
                else waveCtx.lineTo(x, y);

                x += sliceWidth;
            }

            waveCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
            waveCtx.stroke();
        }

        render();
    }

    // Render Spectrogram Canvas Heatmap
    function renderSpectrogram(matrix) {
        if (!matrix || !matrix.length) return;
        const rows = matrix.length;
        const cols = matrix[0].length;

        spectrogramCanvas.width = cols * 2;
        spectrogramCanvas.height = rows * 2;

        const imgData = specCtx.createImageData(spectrogramCanvas.width, spectrogramCanvas.height);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = matrix[r][c];
                // Jet heatmap color mapping: val (0-255)
                const red = val;
                const green = Math.sin((val / 255) * Math.PI) * 255;
                const blue = 255 - val;

                for (let dx = 0; dx < 2; dx++) {
                    for (let dy = 0; dy < 2; dy++) {
                        const px = (c * 2 + dx);
                        const py = ((rows - 1 - r) * 2 + dy); // Flip vertically so high freq is top
                        const index = (py * spectrogramCanvas.width + px) * 4;

                        imgData.data[index] = red;
                        imgData.data[index + 1] = green;
                        imgData.data[index + 2] = blue;
                        imgData.data[index + 3] = 255;
                    }
                }
            }
        }
        specCtx.putImageData(imgData, 0, 0);
    }
});
