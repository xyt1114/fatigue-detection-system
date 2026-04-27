let selectedFile = null
let latestResult = null
let currentPreviewObjectUrl = null

const fileInput = document.getElementById("fileInput")
const chooseBtn = document.getElementById("chooseBtn")
const dropZone = document.getElementById("dropZone")
const uploadForm = document.getElementById("uploadForm")
const uploadBtn = document.getElementById("uploadBtn")
const downloadBtn = document.getElementById("downloadBtn")
const progressBar = document.getElementById("uploadProgress")
const progressText = document.getElementById("progressText")
const fileMeta = document.getElementById("fileMeta")
const loadingArea = document.getElementById("loadingArea")
const emptyResult = document.getElementById("emptyResult")
const previewArea = document.getElementById("previewArea")
const previewImage = document.getElementById("previewImage")
const previewVideo = document.getElementById("previewVideo")
const resultArea = document.getElementById("resultArea")
const resultImage = document.getElementById("resultImage")
const resultVideo = document.getElementById("resultVideo")
const curveArea = document.getElementById("curveArea")
const curveCanvas = document.getElementById("curveCanvas")
const curveMeta = document.getElementById("curveMeta")
const statusCard = document.getElementById("statusCard")
const fatigueBadge = document.getElementById("fatigueBadge")
const scoreText = document.getElementById("scoreText")
const fatigueIcon = document.getElementById("fatigueIcon")
const fatigueLabel = document.getElementById("fatigueLabel")
const earText = document.getElementById("earText")
const marText = document.getElementById("marText")
const poseText = document.getElementById("poseText")
const reasonText = document.getElementById("reasonText")
const warningAlert = document.getElementById("warningAlert")
const historyList = document.getElementById("historyList")
const uploadRuntimeBadge = document.getElementById("uploadRuntimeBadge")
const toastEl = document.getElementById("uploadToast")
const toastBody = document.getElementById("toastBody")
const toastInstance = toastEl ? new bootstrap.Toast(toastEl) : null

function setRuntimeBadge(mode, ready) {
    if (!uploadRuntimeBadge) {
        return
    }
    const m = String(mode || "rule").toUpperCase()
    uploadRuntimeBadge.className = "badge ms-2"
    if (m === "ML" || m === "CNN") {
        uploadRuntimeBadge.classList.add(ready ? "text-bg-success" : "text-bg-warning")
    } else {
        uploadRuntimeBadge.classList.add("text-bg-secondary")
    }
    uploadRuntimeBadge.textContent = `${m} 模式` + (m !== "RULE" ? (ready ? " (已加载)" : " (未加载)") : "")
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

function bytesToSize(size) {
    if (size < 1024) return `${size}B`
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}KB`
    return `${(size / 1024 / 1024).toFixed(2)}MB`
}

function setProgress(percent) {
    const value = Math.max(0, Math.min(100, Math.floor(percent)))
    progressBar.style.width = `${value}%`
    progressBar.setAttribute("aria-valuenow", String(value))
    progressText.textContent = `${value}%`
}

function resetPreview() {
    if (currentPreviewObjectUrl) {
        URL.revokeObjectURL(currentPreviewObjectUrl)
        currentPreviewObjectUrl = null
    }
    previewVideo.pause()
    previewArea.classList.add("d-none")
    previewImage.classList.add("d-none")
    previewVideo.classList.add("d-none")
    previewImage.removeAttribute("src")
    previewVideo.removeAttribute("src")
    previewVideo.load()
}

function canPreviewVideo(file, ext) {
    const mimeCandidates = [
        file.type,
        ext === "mov" ? "video/quicktime" : "",
        `video/${ext}`,
    ].filter(Boolean)
    for (const mime of mimeCandidates) {
        if (previewVideo.canPlayType(mime)) {
            return true
        }
    }
    return ["mp4", "webm", "ogg"].includes(ext)
}

function handleFileSelect(event) {
    const file = event?.target?.files?.[0] || event?.dataTransfer?.files?.[0]
    if (!file) {
        return
    }
    const ext = file.name.toLowerCase().split(".").pop()
    const imageExt = ["jpg", "jpeg", "png", "bmp"]
    const videoExt = ["mp4", "avi", "mov", "mkv"]
    if (![...imageExt, ...videoExt].includes(ext)) {
        showToast("文件类型不支持，请选择图片或视频")
        return
    }
    if (imageExt.includes(ext) && file.size > 10 * 1024 * 1024) {
        showToast("图片大小不能超过10MB")
        return
    }
    if (videoExt.includes(ext) && file.size > 100 * 1024 * 1024) {
        showToast("视频大小不能超过100MB")
        return
    }
    selectedFile = file
    setProgress(0)
    fileMeta.textContent = `已选择：${file.name} · ${bytesToSize(file.size)}`
    previewArea.classList.remove("d-none")
    if (currentPreviewObjectUrl) {
        URL.revokeObjectURL(currentPreviewObjectUrl)
        currentPreviewObjectUrl = null
    }
    const objectUrl = URL.createObjectURL(file)
    currentPreviewObjectUrl = objectUrl
    if (imageExt.includes(ext)) {
        previewImage.src = objectUrl
        previewImage.classList.remove("d-none")
        previewVideo.classList.add("d-none")
        previewVideo.pause()
        previewVideo.removeAttribute("src")
        previewVideo.load()
    } else {
        previewImage.classList.add("d-none")
        if (!canPreviewVideo(file, ext)) {
            previewVideo.classList.add("d-none")
            showToast("该视频编码浏览器可能不支持预览，可直接上传检测", "warning")
            return
        }
        previewVideo.onerror = () => {
            previewVideo.classList.add("d-none")
            showToast("视频无法在当前浏览器预览，可直接上传检测", "warning")
        }
        previewVideo.onloadeddata = () => {
            previewVideo.classList.remove("d-none")
        }
        previewVideo.src = objectUrl
        previewVideo.load()
    }
}

function inferWarning(fatigueLevel) {
    if (fatigueLevel === "severe_fatigue") {
        return "emergency"
    }
    if (fatigueLevel === "fatigue") {
        return "warning"
    }
    return "normal"
}

function drawCurve(curves, summary) {
    if (!curveCanvas) {
        return
    }
    const ctx = curveCanvas.getContext("2d")
    const width = curveCanvas.clientWidth || 860
    const height = curveCanvas.height || 220
    curveCanvas.width = width
    ctx.clearRect(0, 0, width, height)
    const times = (curves && curves.times) || []
    const scoreRaw = (curves && curves.score) || []
    const earRaw = (curves && curves.ear) || []
    const marRaw = (curves && curves.mar) || []
    if (!times.length) {
        ctx.fillStyle = "#6b7280"
        ctx.font = "14px sans-serif"
        ctx.fillText("暂无曲线数据", 16, 28)
        return
    }

    const smooth = (values, windowSize) => {
        if (!values.length) return []
        const half = Math.floor(windowSize / 2)
        return values.map((_, idx) => {
            let sum = 0
            let count = 0
            for (let i = Math.max(0, idx - half); i <= Math.min(values.length - 1, idx + half); i += 1) {
                sum += Number(values[i] || 0)
                count += 1
            }
            return count ? sum / count : 0
        })
    }

    const maxPoints = 240
    const step = Math.max(1, Math.ceil(times.length / maxPoints))
    const sampledTimes = []
    const sampledScoreRaw = []
    const sampledEarRaw = []
    const sampledMarRaw = []
    for (let i = 0; i < times.length; i += step) {
        sampledTimes.push(times[i])
        sampledScoreRaw.push(scoreRaw[i] || 0)
        sampledEarRaw.push(earRaw[i] || 0)
        sampledMarRaw.push(marRaw[i] || 0)
    }

    const score = smooth(sampledScoreRaw, 7).map(v => Math.max(0, Math.min(100, v)))
    const ear = smooth(sampledEarRaw, 7).map(v => Math.max(0, Math.min(1, v)))
    const mar = smooth(sampledMarRaw, 7).map(v => Math.max(0, Math.min(1, v)))

    const x0 = 42
    const y0 = 12
    const chartW = width - 60
    const chartH = height - 40
    const maxTime = Math.max(sampledTimes[sampledTimes.length - 1], 1)
    const toX = t => x0 + (t / maxTime) * chartW
    const toY = v => y0 + chartH - (Math.max(0, Math.min(100, v)) / 100) * chartH

    ;[0, 25, 50, 75, 100].forEach(tick => {
        const y = toY(tick)
        ctx.strokeStyle = tick === 0 ? "#cbd5e1" : "#e5e7eb"
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x0, y)
        ctx.lineTo(x0 + chartW, y)
        ctx.stroke()
    })
    ctx.strokeStyle = "#cbd5e1"
    ctx.strokeRect(x0, y0, chartW, chartH)

    ;((summary && summary.fatigue_segments) || []).forEach(seg => {
        const left = toX(seg.start)
        const right = toX(seg.end)
        ctx.fillStyle = seg.level === "severe_fatigue" ? "rgba(239,68,68,0.16)" : "rgba(245,158,11,0.16)"
        ctx.fillRect(left, y0, Math.max(2, right - left), chartH)
    })
    const drawLine = (values, color) => {
        if (!values.length) return
        ctx.beginPath()
        values.forEach((val, idx) => {
            const x = toX(sampledTimes[idx] || 0)
            const y = toY(val)
            if (idx === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
        })
        ctx.strokeStyle = color
        ctx.lineWidth = color === "#ef4444" ? 2.6 : 1.6
        ctx.stroke()
    }
    drawLine(score, "#ef4444")
    drawLine(ear.map(v => v * 100), "#2563eb")
    drawLine(mar.map(v => v * 100), "#22c55e")
    ctx.fillStyle = "#374151"
    ctx.font = "12px sans-serif"
    ctx.fillText("红:Score(0-100)  蓝:EARx100  绿:MARx100", x0 + 8, height - 8)
}

function displayResult(data) {
    latestResult = data
    emptyResult.classList.add("d-none")
    resultArea.classList.remove("d-none")
    const isVideo = data.mode === "video" || Boolean(data.processed_video_url)
    if (isVideo) {
        if (!data.processed_video_url) {
            showToast("视频结果缺少处理后视频地址，请重试", "danger")
            return
        }
        resultImage.classList.add("d-none")
        resultVideo.classList.remove("d-none")
        resultVideo.src = data.processed_video_url
        resultVideo.load()
        curveArea.classList.remove("d-none")
        drawCurve(data.curves || {}, data.summary || {})
        if (curveMeta) {
            const summary = data.summary || {}
            curveMeta.textContent = `时长 ${Number(summary.duration_sec || 0).toFixed(2)}s，帧数 ${summary.frame_count || 0}，峰值风险 ${summary.max_score || 0}`
        }
    } else {
        resultVideo.classList.add("d-none")
        resultVideo.removeAttribute("src")
        resultVideo.load()
        curveArea.classList.add("d-none")
        if (data.image_with_landmarks) {
            resultImage.src = `data:image/jpeg;base64,${data.image_with_landmarks}`
        }
        resultImage.classList.remove("d-none")
    }
    const level = data.fatigue_level || "alert"
    statusCard.classList.remove("level-alert", "level-fatigue", "level-severe_fatigue")
    statusCard.classList.add(`level-${level}`)
    fatigueBadge.className = "badge rounded-pill px-3 py-2"
    if (level === "severe_fatigue") {
        fatigueBadge.classList.add("text-bg-danger")
        fatigueIcon.className = "bi bi-exclamation-triangle-fill"
    } else if (level === "fatigue") {
        fatigueBadge.classList.add("text-bg-warning")
        fatigueIcon.className = "bi bi-exclamation-circle-fill"
    } else {
        fatigueBadge.classList.add("text-bg-success")
        fatigueIcon.className = "bi bi-emoji-smile"
    }
    fatigueLabel.textContent = level
    scoreText.textContent = `风险分值: ${data.score ?? 0}`
    earText.textContent = Number(data.ear || 0).toFixed(4)
    marText.textContent = Number(data.mar || 0).toFixed(4)
    const pose = data.head_pose || {}
    poseText.textContent = `pitch=${Number(pose.pitch || 0).toFixed(2)}, yaw=${Number(pose.yaw || 0).toFixed(2)}, roll=${Number(pose.roll || 0).toFixed(2)}`
    reasonText.textContent = Array.isArray(data.reasons) && data.reasons.length ? data.reasons.join(", ") : "无"
    const warningLevel = data.warning_level || inferWarning(level)
    warningAlert.classList.remove("d-none", "alert-success", "alert-warning", "alert-danger", "pulse-warning")
    if (warningLevel === "emergency") {
        warningAlert.classList.add("alert-danger", "pulse-warning")
        warningAlert.textContent = "预警级别：紧急，请立即停车休息"
        playAlertTone(920, 320)
    } else if (warningLevel === "warning") {
        warningAlert.classList.add("alert-warning", "pulse-warning")
        warningAlert.textContent = "预警级别：警告，检测到疲劳迹象"
        playAlertTone(760, 220)
    } else {
        warningAlert.classList.add("alert-success")
        warningAlert.textContent = "预警级别：正常"
    }
    setRuntimeBadge(data.inference_mode || "rule", true)
    pushHistory(data)
    downloadBtn.disabled = false
}

function uploadFile(file) {
    return new Promise((resolve, reject) => {
        const formData = new FormData()
        formData.append("file", file)
        const xhr = new XMLHttpRequest()
        xhr.open("POST", "/api/detect_image/", true)
        xhr.upload.onprogress = (event) => {
            if (!event.lengthComputable) {
                return
            }
            const ratio = (event.loaded / event.total) * 80
            setProgress(ratio)
        }
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                setProgress(100)
                try {
                    resolve(JSON.parse(xhr.responseText))
                } catch (error) {
                    reject(new Error("接口响应解析失败"))
                }
                return
            }
            try {
                const data = JSON.parse(xhr.responseText)
                reject(new Error(data.message || "接口调用失败"))
            } catch (error) {
                reject(new Error(`请求失败(${xhr.status})`))
            }
        }
        xhr.onerror = () => reject(new Error("网络错误，请稍后重试"))
        xhr.send(formData)
    })
}

function playAlertTone(freq, duration) {
    const Ctx = window.AudioContext || window.webkitAudioContext
    if (!Ctx) {
        return
    }
    const ctx = new Ctx()
    const oscillator = ctx.createOscillator()
    const gain = ctx.createGain()
    oscillator.type = "sine"
    oscillator.frequency.value = freq
    oscillator.connect(gain)
    gain.connect(ctx.destination)
    gain.gain.setValueAtTime(0.0001, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.01)
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration / 1000)
    oscillator.start()
    oscillator.stop(ctx.currentTime + duration / 1000)
}

function pushHistory(data) {
    const history = JSON.parse(localStorage.getItem("fatigueHistory") || "[]")
    history.unshift({
        time: new Date().toLocaleString(),
        level: data.fatigue_level || "alert",
        score: data.score ?? 0,
        ear: data.ear ?? 0,
        mar: data.mar ?? 0
    })
    const sliced = history.slice(0, 8)
    localStorage.setItem("fatigueHistory", JSON.stringify(sliced))
    renderHistory(sliced)
}

function renderHistory(history) {
    historyList.innerHTML = ""
    if (!history.length) {
        historyList.innerHTML = '<li class="list-group-item text-muted small">暂无记录</li>'
        return
    }
    history.forEach(item => {
        const li = document.createElement("li")
        li.className = "list-group-item small"
        li.textContent = `${item.time} | ${item.level} | score=${item.score} | EAR=${Number(item.ear).toFixed(3)} | MAR=${Number(item.mar).toFixed(3)}`
        historyList.appendChild(li)
    })
}

function downloadReport() {
    if (!latestResult) {
        showToast("暂无可下载报告")
        return
    }
    const report = {
        generated_at: new Date().toISOString(),
        result: latestResult
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = `fatigue_report_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(a.href)
}

function setLoading(isLoading) {
    if (isLoading) {
        loadingArea.classList.remove("d-none")
        uploadBtn.disabled = true
        return
    }
    loadingArea.classList.add("d-none")
    uploadBtn.disabled = false
}

function bindEvents() {
    chooseBtn.addEventListener("click", () => fileInput.click())
    fileInput.addEventListener("change", handleFileSelect)
    dropZone.addEventListener("click", () => fileInput.click())
    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault()
        dropZone.classList.add("dragover")
    })
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"))
    dropZone.addEventListener("drop", (event) => {
        event.preventDefault()
        dropZone.classList.remove("dragover")
        handleFileSelect(event)
    })
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault()
        if (!selectedFile) {
        showToast("请先选择文件", "warning")
            return
        }
        try {
            setLoading(true)
            const data = await uploadFile(selectedFile)
            if (data.status !== "success") {
                throw new Error(data.message || "检测失败")
            }
            displayResult(data)
            showToast("检测完成", "success")
        } catch (error) {
            showToast(error.message || "检测失败", "danger")
        } finally {
            setLoading(false)
        }
    })
    downloadBtn.addEventListener("click", downloadReport)
}

function init() {
    resetPreview()
    setProgress(0)
    loadRuntimeStatus()
    renderHistory(JSON.parse(localStorage.getItem("fatigueHistory") || "[]"))
    bindEvents()
}

document.addEventListener("DOMContentLoaded", init)
