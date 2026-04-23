// static/js/rebuild.js
(function() {
    const detectFolderInput = document.getElementById('detectFolder');
    const mappingFileInput = document.getElementById('mappingFile');
    const originalFolderInput = document.getElementById('originalFolder');
    const outputFolderDisplay = document.getElementById('outputFolderDisplay');
    const nmsSlider = document.getElementById('nmsSlider');
    const nmsValueSpan = document.getElementById('nmsValue');
    const startBtn = document.getElementById('startRebuildBtn');
    const downloadBtn = document.getElementById('downloadResultBtn');
    const backBtn = document.getElementById('backBtn');
    const logBox = document.getElementById('logBox');

    let outputFolder = null;

    function addLog(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logBox.appendChild(entry);
        logBox.scrollTop = logBox.scrollHeight;
    }

    // 获取 URL 参数和 sessionStorage 中的数据
    const urlParams = new URLSearchParams(window.location.search);
    const detectFromUrl = urlParams.get('detect');
    const mappingFromUrl = urlParams.get('mapping');
    const originalFromUrl = urlParams.get('original');
    const timestampFromUrl = urlParams.get('timestamp');
    const workRootFromUrl = urlParams.get('workRoot');

    let detectFolder = detectFromUrl || sessionStorage.getItem('detect_folder') || '';
    let mappingFile = mappingFromUrl || sessionStorage.getItem('crop_mapping_file') || '';
    let originalFolder = originalFromUrl || sessionStorage.getItem('crop_src_folder') || '';
    let timestamp = timestampFromUrl || sessionStorage.getItem('crop_timestamp') || '';
    let workRoot = workRootFromUrl || sessionStorage.getItem('crop_work_root') || '';

    if (detectFolder) detectFolderInput.value = detectFolder;
    if (mappingFile) mappingFileInput.value = mappingFile;
    if (originalFolder) originalFolderInput.value = originalFolder;

    // 自动生成重建输出目录（位于项目根目录下的“重建结果”文件夹）
    if (workRoot) {
        outputFolder = workRoot + '\\重建结果';
        outputFolderDisplay.value = outputFolder;
    } else if (detectFolder) {
        // 兼容旧逻辑（无 workRoot 时回退到上一级目录加“重建结果”）
        let parent = detectFolder.replace(/\\+$/, '');
        let lastSlash = parent.lastIndexOf('\\');
        if (lastSlash !== -1) {
            parent = parent.substring(0, lastSlash);
            outputFolder = parent + '\\重建结果';
            outputFolderDisplay.value = outputFolder;
        }
    }

    // NMS 滑块显示
    nmsSlider.addEventListener('input', () => {
        nmsValueSpan.textContent = parseFloat(nmsSlider.value).toFixed(2);
    });

    // 开始重建
    startBtn.addEventListener('click', async () => {
        const detect = detectFolderInput.value.trim();
        const mapping = mappingFileInput.value.trim();
        const original = originalFolderInput.value.trim();
        const output = outputFolder;

        if (!detect || !mapping || !original || !output) {
            addLog('❌ 缺少必要参数，请确保已完成裁剪和检测步骤', 'error');
            alert('缺少必要参数，请返回上一步重新执行');
            return;
        }

        const nms = parseFloat(nmsSlider.value);

        startBtn.disabled = true;
        downloadBtn.disabled = true;
        addLog('开始重建任务...', 'info');

        try {
            const response = await fetch('/api/rebuild', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detect_result_folder: detect,
                    mapping_file: mapping,
                    original_img_folder: original,
                    output_folder: output,
                    nms_iou: nms
                })
            });
            const result = await response.json();

            if (result.ok) {
                addLog(`✅ 重建完成！`, 'success');
                addLog(`输出目录: ${result.output_folder}`, 'success');
                addLog(`处理看台数: ${result.total_stands}`, 'success');
                addLog(`总检测人数: ${result.total_persons}`, 'success');
                downloadBtn.disabled = false;
                outputFolder = result.output_folder;
                outputFolderDisplay.value = outputFolder;
            } else {
                addLog(`❌ 重建失败: ${result.msg}`, 'error');
                startBtn.disabled = false;
            }
        } catch (error) {
            addLog(`❌ 请求错误: ${error.message}`, 'error');
            startBtn.disabled = false;
        }
    });

    // 下载结果（打包为 zip）
    downloadBtn.addEventListener('click', async () => {
        if (!outputFolder) {
            alert('没有可下载的结果，请先完成重建');
            return;
        }
        addLog('正在打包下载...', 'info');
        try {
            const response = await fetch('/api/download_result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ output_folder: outputFolder })
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'detection_result.zip';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                addLog('✅ 下载已开始', 'success');
            } else {
                const err = await response.json();
                addLog(`❌ 下载失败: ${err.msg}`, 'error');
            }
        } catch (error) {
            addLog(`❌ 下载请求错误: ${error.message}`, 'error');
        }
    });

    // 返回检测页面
    backBtn.addEventListener('click', () => {
        window.location.href = '/detect';
    });
})();