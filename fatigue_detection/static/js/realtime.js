let mediaStream = null
let detectionInterval = null
let isSending = false
let currentWarningLevel = "normal"
let lastWarningSoundAt = 0
let currentSessionId = null
let audioContextRef = null
const warningCooldownMs = 2200
const emergencyCooldownMs = 1200
const DETECTION_FPS = 5
const FRAME_INTERVAL = 1000 / DETECTION_FPS
const SEND_DIFF_THRESHOLD = 2.2
let lastFrameSample = null
let browserFaceDetector = null

const videoEl = document.getElementById("video")
const overlayCanvas = document.getElementById("overlayCanvas")
const overlayCtx = overlayCanvas.getContext("2d")
const captureCanvas = document.getElementById("captureCanvas")
const captureCtx = captureCanvas.getContext("2d")
const cameraSelect = document.getElementById("cameraSelect")
const startBtn = document.getElementById("startBtn")
const stopBtn = document.getElementById("stopBtn")
const snapshotBtn = document.getElementById("snapshotBtn")
const cameraError = document.getElementById("cameraError")
const fatigueLevelText = document.getElementById("fatigueLevelText")
const fatigueLevelIcon = document.getElementById("fatigueLevelIcon")
const fatigueLevelLabel = document.getElementById("fatigueLevelLabel")
const earValue = document.getElementById("earValue")
const marValue = document.getElementById("marValue")
const earProgress = document.getElementById("earProgress")
const marProgress = document.getElementById("marProgress")
const pitchValue = document.getElementById("pitchValue")
const yawValue = document.getElementById("yawValue")
const rollValue = document.getElementById("rollValue")
const warningIndicator = document.getElementById("warningIndicator")
const warningIcon = document.getElementById("warningIcon")
const warningLabel = document.getElementById("warningLabel")
const realtimeRuntimeBadge = document.getElementById("realtimeRuntimeBadge")
const frameCountEl = document.getElementById("frameCount")
const reasonValue = document.getElementById("reasonValue")
const saveConfigBtn = document.getElementById("saveConfigBtn")
const refreshConfigBtn = document.getElementById("refreshConfigBtn")
const resetConfigBtn = document.getElementById("resetConfigBtn")
const toastEl = document.getElementById("realtimeToast")
const toastBody = document.getElementById("realtimeToastBody")
const toastInstance = toastEl ? new bootstrap.Toast(toastEl) : null

function setRuntimeBadge(mode, ready) {
    if (!realtimeRuntimeBadge) {
        return
    }
    const m = String(mode || "rule").toUpperCase()
    realtimeRuntimeBadge.className = "badge ms-2"
    if (m === "ML" || m === "CNN") {
        realtimeRuntimeBadge.classList.add(ready ? "text-bg-success" : "text-bg-warning")
    } else {
        realtimeRuntimeBadge.classList.add("text-bg-secondary")
    }
    realtimeRuntimeBadge.textContent = `${m} 模式` + (m !== "RULE" ? (ready ? " (已加载)" : " (未加载)") : "")
}

async function loadRuntimeStatus() {
    try {
        const resp = await fetch("/api/get_config/")
        const data = await resp.json()
        if (!resp.ok || data.status !== "success") {
            throw new Error("读取运行状态失败")
        }
        const m = String(data.classifier_mode || "rule").toUpperCase()
        const isReady = m === "CNN" ? Boolean(data.cnn_model_ready) : Boolean(data.ml_model_ready)
        setRuntimeBadge(data.classifier_mode, isReady)
    } catch (error) {
        setRuntimeBadge("rule", false)
    }
}

function showToast(message, level = "info") {
    if (!toastInstance || !toastBody) {
        return
    }
    toastEl.classList.remove("toast-info", "toast-success", "toast-warning", "toast-danger")
    toastEl.classList.add(`toast-${level}`)
    toastBody.textContent = message
    toastInstance.show()
}

function setCameraError(message) {
    if (!message) {
        cameraError.classList.add("d-none")
        cameraError.textContent = ""
        return
    }
    cameraError.classList.remove("d-none")
    cameraError.textContent = message
}

function resetStatusUI() {
    fatigueLevelText.className = "fatigue-level level-alert d-inline-flex align-items-center gap-2"
    fatigueLevelIcon.className = "bi bi-emoji-smile"
    fatigueLevelLabel.textContent = "ALERT"
    earValue.textContent = "0.0000"
    marValue.textContent = "0.0000"
    earProgress.style.width = "0%"
    marProgress.style.width = "0%"
    pitchValue.textContent = "0.00"
    yawValue.textContent = "0.00"
    rollValue.textContent = "0.00"
    warningIndicator.className = "warning-indicator level-normal d-inline-flex align-items-center gap-2"
    warningIcon.className = "bi bi-shield-check"
    warningLabel.textContent = "NORMAL"
    frameCountEl.textContent = "0"
    reasonValue.textContent = "无"
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)
}

function syncCanvasSize() {
    const width = videoEl.videoWidth || 640
    const height = videoEl.videoHeight || 360
    if (overlayCanvas.width !== width || overlayCanvas.height !== height) {
        overlayCanvas.width = width
        overlayCanvas.height = height
    }
    if (captureCanvas.width !== width || captureCanvas.height !== height) {
        captureCanvas.width = width
        captureCanvas.height = height
    }
}

function buildFrameSample() {
    const sampleCanvas = document.createElement("canvas")
    sampleCanvas.width = 64
    sampleCanvas.height = 36
    const sampleCtx = sampleCanvas.getContext("2d")
    sampleCtx.drawImage(videoEl, 0, 0, sampleCanvas.width, sampleCanvas.height)
    const pixels = sampleCtx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data
    let sum = 0
    for (let i = 0; i < pixels.length; i += 16) {
        sum += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
    }
    return sum / (pixels.length / 16)
}

function shouldSendByMotion() {
    const currentSample = buildFrameSample()
    if (lastFrameSample === null) {
        lastFrameSample = currentSample
        return true
    }
    const diff = Math.abs(currentSample - lastFrameSample)
    lastFrameSample = currentSample
    return diff >= SEND_DIFF_THRESHOLD
}

function encodeFrameForUpload() {
    const uploadCanvas = document.createElement("canvas")
    const maxSize = 640
    const w = videoEl.videoWidth || 640
    const h = videoEl.videoHeight || 360
    const scale = Math.min(1, maxSize / Math.max(w, h))
    uploadCanvas.width = Math.max(1, Math.floor(w * scale))
    uploadCanvas.height = Math.max(1, Math.floor(h * scale))
    const uploadCtx = uploadCanvas.getContext("2d")
    uploadCtx.drawImage(videoEl, 0, 0, uploadCanvas.width, uploadCanvas.height)
    return uploadCanvas.toDataURL("image/jpeg", 0.68)
}

async function shouldSendByFace() {
    if (!browserFaceDetector) {
        return true
    }
    try {
        const faces = await browserFaceDetector.detect(videoEl)
        return Array.isArray(faces) && faces.length > 0
    } catch (error) {
        return true
    }
}

async function stopCurrentStream() {
    if (!mediaStream) {
        return
    }
    mediaStream.getTracks().forEach(track => track.stop())
    mediaStream = null
}

async function initWebcam(deviceId = null) {
    try {
        setCameraError("")
        await stopCurrentStream()
        const constraints = {
            audio: false,
            video: deviceId
                ? { deviceId: { exact: deviceId } }
                : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
        }
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
        videoEl.srcObject = mediaStream
        await videoEl.play()
        syncCanvasSize()
        const devices = await navigator.mediaDevices.enumerateDevices()
        const cameras = devices.filter(item => item.kind === "videoinput")
        cameraSelect.innerHTML = ""
        cameras.forEach((camera, index) => {
            const option = document.createElement("option")
            option.value = camera.deviceId
            option.textContent = camera.label || `摄像头 ${index + 1}`
            cameraSelect.appendChild(option)
        })
        const activeTrack = mediaStream.getVideoTracks()[0]
        const activeDeviceId = activeTrack?.getSettings()?.deviceId
        if (activeDeviceId) {
            cameraSelect.value = activeDeviceId
        }
        if (!cameras.length) {
            setCameraError("未找到可用摄像头")
        }
    } catch (error) {
        const message = error?.name === "NotAllowedError" ? "未授予摄像头权限" : "摄像头初始化失败，请检查设备与浏览器权限"
        setCameraError(message)
        showToast(message)
    }
}

function drawLandmarks(canvas, video, landmarks, fatigueStatus) {
    if (!canvas || !video) {
        return
    }
    const ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    let color = "#1f8f45"
    if (fatigueStatus === "fatigue") {
        color = "#b07b00"
    } else if (fatigueStatus === "severe_fatigue") {
        color = "#bb1f2f"
    }
    ctx.lineWidth = 4
    ctx.strokeStyle = color
    ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4)
    ctx.font = "18px sans-serif"
    ctx.fillStyle = color
    ctx.fillText(`STATUS: ${String(fatigueStatus || "alert").toUpperCase()}`, 14, 30)
    if (Array.isArray(landmarks)) {
        ctx.fillStyle = "#00ff8a"
        landmarks.forEach(point => {
            const x = Number(point.x || point[0] || 0)
            const y = Number(point.y || point[1] || 0)
            ctx.beginPath()
            ctx.arc(x, y, 2.2, 0, Math.PI * 2)
            ctx.fill()
        })
    }
}

function updateWarningUI(level) {
    warningIndicator.className = "warning-indicator d-inline-flex align-items-center gap-2"
    if (level === "emergency") {
        warningIndicator.classList.add("level-emergency", "pulse-warning")
        warningIcon.className = "bi bi-bell-fill"
        warningLabel.textContent = "EMERGENCY"
    } else if (level === "warning") {
        warningIndicator.classList.add("level-warning")
        warningIndicator.classList.remove("pulse-warning")
        warningIcon.className = "bi bi-exclamation-circle-fill"
        warningLabel.textContent = "WARNING"
    } else {
        warningIndicator.classList.add("level-normal")
        warningIndicator.classList.remove("pulse-warning")
        warningIcon.className = "bi bi-shield-check"
        warningLabel.textContent = "NORMAL"
    }
}

function ensureAudioContext() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    if (!AudioCtx) {
        return null
    }
    if (!audioContextRef) {
        audioContextRef = new AudioCtx()
    }
    if (audioContextRef.state === "suspended") {
        audioContextRef.resume().catch(() => null)
    }
    return audioContextRef
}

function playBeep(frequency, durationMs) {
    const ctx = ensureAudioContext()
    if (!ctx) {
        return
    }
    const oscillator = ctx.createOscillator()
    const gain = ctx.createGain()
    oscillator.connect(gain)
    gain.connect(ctx.destination)
    oscillator.type = "square"
    oscillator.frequency.value = frequency
    gain.gain.setValueAtTime(0.0001, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.16, ctx.currentTime + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + durationMs / 1000)
    oscillator.start()
    oscillator.stop(ctx.currentTime + durationMs / 1000)
}

function playWarning(level) {
    if (level !== "warning" && level !== "emergency") {
        return
    }
    const now = Date.now()
    const cooldown = level === "emergency" ? emergencyCooldownMs : warningCooldownMs
    if (now - lastWarningSoundAt < cooldown) {
        return
    }
    lastWarningSoundAt = now
    if (level === "emergency") {
        playBeep(940, 280)
        return
    }
    playBeep(760, 220)
}

function handleFrameResult(data) {
    const level = data.fatigue_level || "alert"
    fatigueLevelText.className = `fatigue-level level-${level} d-inline-flex align-items-center gap-2`
    fatigueLevelLabel.textContent = String(level).toUpperCase()
    fatigueLevelIcon.className = level === "severe_fatigue" ? "bi bi-exclamation-triangle-fill" : (level === "fatigue" ? "bi bi-exclamation-circle-fill" : "bi bi-emoji-smile")
    const ear = Number(data.ear || 0)
    const mar = Number(data.mar || 0)
    earValue.textContent = ear.toFixed(4)
    marValue.textContent = mar.toFixed(4)
    earProgress.style.width = `${Math.min(100, Math.max(0, ear * 300))}%`
    marProgress.style.width = `${Math.min(100, Math.max(0, mar * 120))}%`
    const pose = data.head_pose || {}
    pitchValue.textContent = Number(pose.pitch || 0).toFixed(2)
    yawValue.textContent = Number(pose.yaw || 0).toFixed(2)
    rollValue.textContent = Number(pose.roll || 0).toFixed(2)
    const reasons = Array.isArray(data.reasons) ? data.reasons : []
    reasonValue.textContent = reasons.length ? reasons.join(", ") : "无"
    const warningLevel = data.warning_level || "normal"
    frameCountEl.textContent = String(data.frame_count || 0)
    updateWarningUI(warningLevel)
    drawLandmarks(overlayCanvas, videoEl, data.landmarks, level)
    playWarning(warningLevel)
    setRuntimeBadge(data.inference_mode || "rule", true)
    currentWarningLevel = warningLevel
}

async function detectFrameOnce() {
    if (isSending || !mediaStream || videoEl.readyState < 2) {
        return
    }
    if (!shouldSendByMotion()) {
        return
    }
    if (!(await shouldSendByFace())) {
        return
    }
    isSending = true
    try {
        syncCanvasSize()
        captureCtx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height)
        const frameBase64 = encodeFrameForUpload()
        const response = await fetch("/api/detect_frame/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameBase64, persist: true, session_id: currentSessionId })
        })
        const result = await response.json()
        if (!response.ok || result.status !== "success") {
            throw new Error(result.message || "帧检测失败")
        }
        if (result.session_id) {
            currentSessionId = result.session_id
        }
        handleFrameResult(result)
    } catch (error) {
        showToast(error.message || "实时检测失败")
    } finally {
        isSending = false
    }
}

function startDetection() {
    if (detectionInterval || !mediaStream) {
        return
    }
    ensureAudioContext()
    startBtn.disabled = true
    stopBtn.disabled = false
    snapshotBtn.disabled = false
    detectionInterval = setInterval(detectFrameOnce, FRAME_INTERVAL)
    showToast("已开始实时检测")
}

function stopDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval)
        detectionInterval = null
    }
    startBtn.disabled = false
    stopBtn.disabled = true
    snapshotBtn.disabled = true
    currentWarningLevel = "normal"
    currentSessionId = null
    resetStatusUI()
}

function captureSnapshot() {
    if (!mediaStream || videoEl.readyState < 2) {
        showToast("摄像头未准备好")
        return
    }
    syncCanvasSize()
    const saveCanvas = document.createElement("canvas")
    saveCanvas.width = overlayCanvas.width
    saveCanvas.height = overlayCanvas.height
    const ctx = saveCanvas.getContext("2d")
    ctx.drawImage(videoEl, 0, 0, saveCanvas.width, saveCanvas.height)
    ctx.drawImage(overlayCanvas, 0, 0, saveCanvas.width, saveCanvas.height)
    const url = saveCanvas.toDataURL("image/png")
    const link = document.createElement("a")
    link.href = url
    link.download = `realtime_snapshot_${Date.now()}.png`
    link.click()
}

function bindEvents() {
    startBtn.addEventListener("click", startDetection)
    stopBtn.addEventListener("click", stopDetection)
    snapshotBtn.addEventListener("click", captureSnapshot)
    refreshConfigBtn.addEventListener("click", async () => {
        try {
            await window.ConfigUI.loadConfig(showToast)
        } catch (error) {
            showToast(error.message || "读取配置失败")
        }
    })
    saveConfigBtn.addEventListener("click", async () => {
        try {
            await window.ConfigUI.saveConfig(showToast)
        } catch (error) {
            showToast(error.message || "保存配置失败")
        }
    })
    resetConfigBtn.addEventListener("click", async () => {
        try {
            await window.ConfigUI.resetConfig(showToast)
        } catch (error) {
            showToast(error.message || "恢复默认失败")
        }
    })
    cameraSelect.addEventListener("change", async event => {
        const selected = event.target.value
        const wasDetecting = Boolean(detectionInterval)
        stopDetection()
        await initWebcam(selected)
        if (wasDetecting) {
            startDetection()
        }
    })
    window.addEventListener("resize", syncCanvasSize)
}

async function bootstrapRealtime() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError("当前浏览器不支持摄像头访问")
        return
    }
    resetStatusUI()
    if ("FaceDetector" in window) {
        try {
            browserFaceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 })
        } catch (error) {
            browserFaceDetector = null
        }
    }
    bindEvents()
    await initWebcam()
    await loadRuntimeStatus()
    try {
        await window.ConfigUI.loadConfig()
    } catch (error) {
        showToast(error.message || "读取配置失败")
    }
}

document.addEventListener("DOMContentLoaded", bootstrapRealtime)
