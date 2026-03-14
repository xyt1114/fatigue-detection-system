(function () {
    const defaults = {
        ear_threshold: 0.25,
        mar_threshold: 0.6,
        pitch_threshold: 30,
        warning_frame_count: 3,
        emergency_frame_count: 5
    }

    const earInput = document.getElementById("earThresholdInput")
    const marInput = document.getElementById("marThresholdInput")
    const pitchInput = document.getElementById("pitchThresholdInput")
    const warningInput = document.getElementById("warningFrameCountInput")
    const emergencyInput = document.getElementById("emergencyFrameCountInput")

    function fillConfig(config) {
        if (!earInput || !marInput || !pitchInput || !warningInput || !emergencyInput) {
            return
        }
        earInput.value = Number(config.ear_threshold).toFixed(2)
        marInput.value = Number(config.mar_threshold).toFixed(2)
        pitchInput.value = Number(config.pitch_threshold).toFixed(0)
        warningInput.value = Number(config.warning_frame_count).toFixed(0)
        emergencyInput.value = Number(config.emergency_frame_count).toFixed(0)
    }

    function collectConfig() {
        return {
            ear_threshold: Number(earInput.value),
            mar_threshold: Number(marInput.value),
            pitch_threshold: Number(pitchInput.value),
            warning_frame_count: Number(warningInput.value),
            emergency_frame_count: Number(emergencyInput.value)
        }
    }

    function validateConfig(config) {
        if (config.ear_threshold < 0.1 || config.ear_threshold > 0.4) {
            throw new Error("EAR阈值范围应在0.1-0.4")
        }
        if (config.mar_threshold < 0.3 || config.mar_threshold > 0.8) {
            throw new Error("MAR阈值范围应在0.3-0.8")
        }
        if (config.pitch_threshold < 10 || config.pitch_threshold > 60) {
            throw new Error("Pitch阈值范围应在10-60")
        }
        if (config.warning_frame_count < 1 || config.warning_frame_count > 30) {
            throw new Error("预警帧阈值范围应在1-30")
        }
        if (config.emergency_frame_count < 1 || config.emergency_frame_count > 60) {
            throw new Error("紧急帧阈值范围应在1-60")
        }
        if (config.emergency_frame_count < config.warning_frame_count) {
            throw new Error("紧急帧阈值不能小于预警帧阈值")
        }
    }

    async function loadConfig(notify) {
        const response = await fetch("/api/get_config/")
        const data = await response.json()
        if (!response.ok || data.status !== "success") {
            throw new Error("读取配置失败")
        }
        fillConfig(data.config || defaults)
        if (typeof notify === "function") {
            notify("配置已加载")
        }
        return data.config
    }

    async function saveConfig(notify) {
        const payload = collectConfig()
        validateConfig(payload)
        const response = await fetch("/api/update_config/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        const data = await response.json()
        if (!response.ok || data.status !== "success") {
            throw new Error("保存配置失败")
        }
        fillConfig(data.config || payload)
        if (typeof notify === "function") {
            notify("配置已更新")
        }
        return data.config
    }

    async function resetConfig(notify) {
        fillConfig(defaults)
        return saveConfig(notify)
    }

    window.ConfigUI = {
        loadConfig,
        saveConfig,
        resetConfig,
        defaults
    }
})()
