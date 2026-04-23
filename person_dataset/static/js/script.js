let currentFile = null;
let currentFolderFiles = [];

// 滑块同步
const confSlider = document.getElementById('conf-slider');
const confValue = document.getElementById('conf-value');
const iouSlider = document.getElementById('iou-slider');
const iouValue = document.getElementById('iou-value');

confSlider.addEventListener('input', () => {
    confValue.textContent = confSlider.value;
});
iouSlider.addEventListener('input', () => {
    iouValue.textContent = iouSlider.value;
});

// 图片选择
document.getElementById('btn-img').addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = 'image/*';
    inp.onchange = e => {
        const file = e.target.files[0];
        if (!file) return;
        currentFile = file;
        currentFolderFiles = [];
        const url = URL.createObjectURL(file);
        document.getElementById('detect-img').src = url;
        document.getElementById('detect-img').style.visibility = 'visible';
        document.querySelector('.empty-tip').style.display = 'none';
        const canvas = document.getElementById('canvas-overlay');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result-tbody').innerHTML = '';
        document.getElementById('target-count').textContent = '0个';
        document.getElementById('time-cost').textContent = '0.00s';
        window.allPersons = [];
        document.getElementById('coord-x1').value = 0;
        document.getElementById('coord-y1').value = 0;
        document.getElementById('coord-x2').value = 0;
        document.getElementById('coord-y2').value = 0;

        if(document.getElementById('conf-input')){
            document.getElementById('conf-input').value = '';
        }

        removeProgressBar();
    };
    inp.click();
});

// 文件夹选择
document.getElementById('btn-folder').onclick = function (e) {
    e.preventDefault();
    e.stopImmediatePropagation();

    const inp = document.createElement('input');
    inp.type = 'file';
    inp.webkitdirectory = true;
    inp.accept = 'image/*';

    inp.onchange = function (e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;
        currentFolderFiles = Array.from(files);
        currentFile = null;
        const firstFile = currentFolderFiles[0];
        const url = URL.createObjectURL(firstFile);
        document.getElementById('detect-img').src = url;
        document.getElementById('detect-img').style.visibility = 'visible';
        document.querySelector('.empty-tip').style.display = 'none';
        const canvas = document.getElementById('canvas-overlay');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result-tbody').innerHTML = '';
        document.getElementById('target-count').textContent = '0个';
        document.getElementById('time-cost').textContent = '0.00s';
        window.allPersons = [];
        document.getElementById('coord-x1').value = 0;
        document.getElementById('coord-y1').value = 0;
        document.getElementById('coord-x2').value = 0;
        document.getElementById('coord-y2').value = 0;

        if(document.getElementById('conf-input')){
            document.getElementById('conf-input').value = '';
        }

        removeProgressBar();
    };
    inp.click();
};

// ======================
// 模型选择（修复！可正常弹出）
// ======================
document.getElementById('btn-model').addEventListener('click', async () => {
    const modal = document.getElementById('modelModal');
    const modelList = document.getElementById('model-list');

    try {
        const res = await fetch('/get_models');
        const data = await res.json();
        modelList.innerHTML = '';

        if (!data.models || data.models.length === 0) {
            modelList.innerHTML = '<div class="model-option">暂无模型</div>';
            modal.style.display = 'flex';
            return;
        }

        data.models.forEach(m => {
            const div = document.createElement('div');
            div.className = 'model-option';
            div.textContent = m.name;
            div.onclick = async () => {
                await fetch('/switch_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: m.name })
                });
                alert('模型切换成功：' + m.name);
                modal.style.display = 'none';
            };
            modelList.appendChild(div);
        });
        modal.style.display = 'flex';
    } catch (e) {
        alert('获取模型失败');
        console.error(e);
    }
});

// 点击遮罩关闭弹窗
document.getElementById('modelModal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('modelModal')) {
        document.getElementById('modelModal').style.display = 'none';
    }
});

// ======================
// 开始检测（修复画框 + 支持人头点）
// ======================
document.getElementById('start-detect').addEventListener('click', async () => {
    const isFolderMode = currentFolderFiles.length > 0;
    if (!currentFile && !isFolderMode) {
        alert('请先选择图片或文件夹');
        return;
    }

    const tbody = document.getElementById('result-tbody');
    tbody.innerHTML = '';
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    window.allPersons = [];
    removeProgressBar();

    const progressBar = document.createElement("div");
    progressBar.innerHTML = `
        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 50%;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);
            z-index: 999999;
        ">
            <div style="margin-bottom:8px; font-size:16px;" id="p-text">检测中...</div>
            <div style="width:100%; height:14px; background:#eee; border-radius:7px; overflow:hidden;">
                <div id="p-fill" style="width:0%; height:100%; background:#0096ff; transition:width 0.3s;"></div>
            </div>
        </div>
    `;
    document.body.appendChild(progressBar);
    const pText = progressBar.querySelector("#p-text");
    const pFill = progressBar.querySelector("#p-fill");

    try {
        if (isFolderMode) {
            let totalCount = 0;
            const allResults = [];
            const totalFiles = currentFolderFiles.length;

            for (let i = 0; i < totalFiles; i++) {
                const file = currentFolderFiles[i];
                const per = ((i + 1) / totalFiles) * 100;
                pText.textContent = `检测中：${i+1}/${totalFiles}`;
                pFill.style.width = per + "%";

                const fd = new FormData();
                fd.append('image', file);
                fd.append('conf', confSlider.value);
                fd.append('iou', iouSlider.value);

                const res = await fetch('/detect_single', { method: 'POST', body: fd });
                const data = await res.json();

                allResults.push({ file, data });
                totalCount += data.count || 0;

                if (i === 0) {
                    document.getElementById('detect-img').src = data.image_data;
                    window.allPersons = data.persons || [];
                    renderSingleResult(data, file);
                }
            }

            pText.textContent = `完成！共${totalFiles}张，总人数${totalCount}`;
            pFill.style.width = "100%";
            setTimeout(() => progressBar.remove(), 1600);

            tbody.innerHTML = '';
            allResults.forEach((item, idx) => {
                const { file, data } = item;
                data.persons.forEach((p, i) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${idx+1}-${i+1}</td>
                        <td>${file.name}</td>
                        <td>${p.class_name || 'person'}</td>
                        <td>${p.conf.toFixed(2)}</td>
                        <td>${p.x1},${p.y1},${p.x2},${p.y2}</td>
                    `;
                    tr.onclick = () => {
                        document.getElementById('detect-img').src = data.image_data;
                        window.allPersons = data.persons || [];
                        renderSingleResult(data, file);
                    };
                    tbody.appendChild(tr);
                });
            });

            document.getElementById('target-count').textContent = `${totalCount}个`;
            document.getElementById('stats-content').innerHTML = `<div class="stats-item"><span>行人</span><span class="value">${totalCount}</span></div>`;

        } else {
            pText.textContent = "检测中...";
            pFill.style.width = "80%";

            const fd = new FormData();
            fd.append('image', currentFile);
            fd.append('conf', confSlider.value);
            fd.append('iou', iouSlider.value);

            const res = await fetch('/detect_single', { method: 'POST', body: fd });
            const data = await res.json();
            progressBar.remove();

            document.getElementById('time-cost').textContent = (data.infer_time || 0).toFixed(2) + 's';
            document.getElementById('target-count').textContent = (data.count || 0) + '个';
            window.allPersons = data.persons || [];
            renderSingleResult(data, currentFile);

            document.getElementById('stats-content').innerHTML = `<div class="stats-item"><span>行人</span><span class="value">${data.count || 0}</span></div>`;

            if (data.image_data) {
                document.getElementById('detect-img').src = data.image_data;
            }
        }

    } catch (err) {
        progressBar.remove();
        alert('检测失败');
        console.error(err);
    }
});

// ======================
// 渲染结果（修复：自动缩放坐标，不重叠）
// ======================
function renderSingleResult(data, file) {
    const tbody = document.getElementById('result-tbody');
    tbody.innerHTML = '';
    const persons = data.persons || [];

    persons.forEach((p, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i+1}</td>
            <td>${file.name}</td>
            <td>${p.class_name || 'person'}</td>
            <td>${p.conf.toFixed(2)}</td>
            <td>${p.x1},${p.y1},${p.x2},${p.y2}</td>
        `;
        tr.onclick = () => highlightPerson(i);
        tbody.appendChild(tr);
    });

    // 清空右侧面板
    document.getElementById('coord-x1').value = 0;
    document.getElementById('coord-y1').value = 0;
    document.getElementById('coord-x2').value = 0;
    document.getElementById('coord-y2').value = 0;
    if (document.getElementById('conf-input')) {
        document.getElementById('conf-input').value = '';
    }

    // 自动适配画布
    const img = document.getElementById('detect-img');
    const canvas = document.getElementById('canvas-overlay');
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;

    // 绘制所有框
    drawAllBoxes();
}

// ======================
// 绘制所有目标框（修复！不重叠）
// ======================
function drawAllBoxes() {
    const img = document.getElementById('detect-img');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    const persons = window.allPersons || [];

    persons.forEach(p => {
        const x1 = p.x1 * scaleX;
        const y1 = p.y1 * scaleY;
        const w = (p.x2 - p.x1) * scaleX;
        const h = (p.y2 - p.y1) * scaleY;

        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);
    });
}

// ======================
// 高亮选中目标（修复）
// ======================
function highlightPerson(index) {
    const rows = document.querySelectorAll('#result-tbody tr');
    const persons = window.allPersons || [];
    if (index < 0 || index >= persons.length) return;

    rows.forEach((row, i) => {
        row.style.backgroundColor = i === index ? '#e3f2fd' : '';
        row.style.fontWeight = i === index ? 'bold' : '';
    });

    const p = persons[index];
    document.getElementById('coord-x1').value = p.x1;
    document.getElementById('coord-y1').value = p.y1;
    document.getElementById('coord-x2').value = p.x2;
    document.getElementById('coord-y2').value = p.y2;
    if (document.getElementById('conf-input')) {
        document.getElementById('conf-input').value = p.conf.toFixed(2);
    }

    // 重绘 + 高亮
    drawAllBoxes();
    const img = document.getElementById('detect-img');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;

    const x1 = p.x1 * scaleX;
    const y1 = p.y1 * scaleY;
    const w = (p.x2 - p.x1) * scaleX;
    const h = (p.y2 - p.y1) * scaleY;

    ctx.strokeStyle = '#ff3333';
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, w, h);

    // 标签
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x1, y1 - 24, 140, 24);
    ctx.fillStyle = "#fff";
    ctx.font = "14px Arial";
    ctx.fillText(`person ${p.conf.toFixed(2)}`, x1 + 6, y1 - 6);
}

// 结果保存
document.getElementById('btn-save').addEventListener('click', async () => {
    if (!window.allPersons || window.allPersons.length === 0) {
        alert('暂无检测结果');
        return;
    }
    try {
        const res = await fetch('/save_results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                persons: window.allPersons,
                image_base64: document.getElementById('detect-img').src
            })
        });
        const data = await res.json();
        alert(data.success ? '保存成功' : '保存失败');
    } catch (e) {
        alert('保存异常');
    }
});

function removeProgressBar(){}

// ====================== 批量处理工作流集成 ======================
(function() {
    // 步骤折叠/展开
    const stepHeaders = document.querySelectorAll('.step-header');
    const stepContents = {
        1: document.getElementById('step1-content'),
        2: document.getElementById('step2-content'),
        3: document.getElementById('step3-content')
    };
    stepHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const step = header.dataset.step;
            stepContents[step].classList.toggle('active');
        });
    });

    // DOM 元素
    const srcFolder = document.getElementById('wf-src-folder');
    const dstFolder = document.getElementById('wf-dst-folder');
    const tileSize = document.getElementById('wf-tile-size');
    const overlap = document.getElementById('wf-overlap');
    const filterEmpty = document.getElementById('wf-filter-empty');
    const detectFolder = document.getElementById('wf-detect-folder');
    const confSlider = document.getElementById('wf-conf');
    const confVal = document.getElementById('wf-conf-val');
    const iouSlider = document.getElementById('wf-iou');
    const iouVal = document.getElementById('wf-iou-val');
    const originalFolder = document.getElementById('wf-original-folder');
    const outputFolder = document.getElementById('wf-output-folder');
    const nmsSlider = document.getElementById('wf-nms');
    const nmsVal = document.getElementById('wf-nms-val');

    const runCrop = document.getElementById('wf-run-crop');
    const runDetect = document.getElementById('wf-run-detect');
    const runRebuild = document.getElementById('wf-run-rebuild');
    const runAll = document.getElementById('wf-run-all');
    const logBox = document.getElementById('workflow-log');

    // 滑块显示同步
    confSlider.addEventListener('input', () => confVal.textContent = parseFloat(confSlider.value).toFixed(2));
    iouSlider.addEventListener('input', () => iouVal.textContent = parseFloat(iouSlider.value).toFixed(2));
    nmsSlider.addEventListener('input', () => nmsVal.textContent = parseFloat(nmsSlider.value).toFixed(2));

    // 日志函数
    function addLog(msg, type = 'info') {
        const line = document.createElement('div');
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        line.style.color = type === 'error' ? '#f87171' : (type === 'success' ? '#6ee7b7' : '#e2e8f0');
        logBox.appendChild(line);
        logBox.scrollTop = logBox.scrollHeight;
    }

    // 保存/恢复输入（使用 sessionStorage）
    function saveState() {
        sessionStorage.setItem('wf_src', srcFolder.value);
        sessionStorage.setItem('wf_dst', dstFolder.value);
        sessionStorage.setItem('wf_tile', tileSize.value);
        sessionStorage.setItem('wf_overlap', overlap.value);
        sessionStorage.setItem('wf_filter', filterEmpty.checked);
        sessionStorage.setItem('wf_detect', detectFolder.value);
        sessionStorage.setItem('wf_conf', confSlider.value);
        sessionStorage.setItem('wf_iou', iouSlider.value);
        sessionStorage.setItem('wf_original', originalFolder.value);
        sessionStorage.setItem('wf_output', outputFolder.value);
        sessionStorage.setItem('wf_nms', nmsSlider.value);
    }

    function restoreState() {
        srcFolder.value = sessionStorage.getItem('wf_src') || '';
        dstFolder.value = sessionStorage.getItem('wf_dst') || '';
        tileSize.value = sessionStorage.getItem('wf_tile') || '1280';
        overlap.value = sessionStorage.getItem('wf_overlap') || '200';
        filterEmpty.checked = sessionStorage.getItem('wf_filter') !== 'false';
        detectFolder.value = sessionStorage.getItem('wf_detect') || '';
        confSlider.value = sessionStorage.getItem('wf_conf') || '0.5';
        iouSlider.value = sessionStorage.getItem('wf_iou') || '0.5';
        originalFolder.value = sessionStorage.getItem('wf_original') || '';
        outputFolder.value = sessionStorage.getItem('wf_output') || '';
        nmsSlider.value = sessionStorage.getItem('wf_nms') || '0.4';
        confVal.textContent = parseFloat(confSlider.value).toFixed(2);
        iouVal.textContent = parseFloat(iouSlider.value).toFixed(2);
        nmsVal.textContent = parseFloat(nmsSlider.value).toFixed(2);
        runDetect.disabled = !detectFolder.value;
        runRebuild.disabled = !(detectFolder.value && originalFolder.value && outputFolder.value);
    }

    [srcFolder, dstFolder, tileSize, overlap, filterEmpty, detectFolder, confSlider, iouSlider, originalFolder, outputFolder, nmsSlider]
        .forEach(el => el.addEventListener('change', saveState));

    restoreState();
    stepContents[1].classList.add('active'); // 默认展开第一步

    // 裁剪
    runCrop.addEventListener('click', async () => {
        if (!srcFolder.value.trim() || !dstFolder.value.trim()) {
            alert('请填写源文件夹和输出文件夹');
            return;
        }
        runCrop.disabled = true;
        addLog('开始裁剪...', 'info');
        try {
            const res = await fetch('/api/crop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    src_folder: srcFolder.value.trim(),
                    dst_folder: dstFolder.value.trim(),
                    tile_size: parseInt(tileSize.value),
                    overlap: parseInt(overlap.value),
                    filter_empty: filterEmpty.checked
                })
            });
            const data = await res.json();
            if (data.ok) {
                addLog(`✅ 裁剪完成，生成 ${data.total_tiles} 张小图`, 'success');
                detectFolder.value = dstFolder.value.trim();
                originalFolder.value = srcFolder.value.trim();
                saveState();
                runDetect.disabled = false;
                runRebuild.disabled = !outputFolder.value.trim();
            } else {
                addLog(`❌ 裁剪失败: ${data.msg}`, 'error');
            }
        } catch (e) {
            addLog(`❌ 请求错误: ${e.message}`, 'error');
        } finally {
            runCrop.disabled = false;
        }
    });

    // 检测
    runDetect.addEventListener('click', async () => {
        const folder = detectFolder.value.trim();
        if (!folder) {
            alert('请先完成裁剪步骤');
            return;
        }
        runDetect.disabled = true;
        addLog(`开始检测文件夹: ${folder}`, 'info');
        try {
            const res = await fetch('/detect_folder', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder_path: folder,
                    conf: parseFloat(confSlider.value),
                    iou: parseFloat(iouSlider.value)
                })
            });
            const data = await res.json();
            if (!data.error) {
                addLog(`✅ 检测完成，共 ${data.total} 张图片，检测到 ${data.total_count} 人`, 'success');
                runRebuild.disabled = !(originalFolder.value.trim() && outputFolder.value.trim());
            } else {
                addLog(`❌ 检测失败: ${data.error}`, 'error');
            }
        } catch (e) {
            addLog(`❌ 请求错误: ${e.message}`, 'error');
        } finally {
            runDetect.disabled = false;
        }
    });

    // 重建
    runRebuild.addEventListener('click', async () => {
        const detect = detectFolder.value.trim();
        const original = originalFolder.value.trim();
        const output = outputFolder.value.trim();
        const mapping = dstFolder.value.trim() + '\\crop_mapping.xlsx';
        if (!detect || !original || !output) {
            alert('请确保检测结果、原始大图、输出目录均已填写');
            return;
        }
        runRebuild.disabled = true;
        addLog('开始重建...', 'info');
        try {
            const res = await fetch('/api/rebuild', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    detect_result_folder: detect,
                    mapping_file: mapping,
                    original_img_folder: original,
                    output_folder: output,
                    nms_iou: parseFloat(nmsSlider.value)
                })
            });
            const data = await res.json();
            if (data.ok) {
                addLog(`✅ 重建完成！处理看台数: ${data.total_stands}，总人数: ${data.total_persons}`, 'success');
                addLog(`📁 结果保存至: ${data.output_folder}`, 'success');
            } else {
                addLog(`❌ 重建失败: ${data.msg}`, 'error');
            }
        } catch (e) {
            addLog(`❌ 请求错误: ${e.message}`, 'error');
        } finally {
            runRebuild.disabled = false;
        }
    });

    // 一键执行
    runAll.addEventListener('click', async () => {
        if (!srcFolder.value.trim() || !dstFolder.value.trim()) {
            alert('请先填写裁剪源文件夹和目标文件夹');
            return;
        }
        if (!outputFolder.value.trim()) {
            alert('请填写重建输出文件夹');
            return;
        }
        runAll.disabled = true;
        addLog('===== 开始完整工作流 =====', 'info');

        try {
            // 裁剪
            if (!detectFolder.value) {
                addLog('步骤1：裁剪中...', 'info');
                const cropRes = await fetch('/api/crop', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        src_folder: srcFolder.value.trim(),
                        dst_folder: dstFolder.value.trim(),
                        tile_size: parseInt(tileSize.value),
                        overlap: parseInt(overlap.value),
                        filter_empty: filterEmpty.checked
                    })
                });
                const cropData = await cropRes.json();
                if (!cropData.ok) throw new Error('裁剪失败: ' + cropData.msg);
                detectFolder.value = dstFolder.value.trim();
                originalFolder.value = srcFolder.value.trim();
                saveState();
                addLog(`✅ 裁剪完成，生成 ${cropData.total_tiles} 张小图`, 'success');
            }

            // 检测
            addLog('步骤2：批量检测中...', 'info');
            const detectRes = await fetch('/detect_folder', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder_path: detectFolder.value.trim(),
                    conf: parseFloat(confSlider.value),
                    iou: parseFloat(iouSlider.value)
                })
            });
            const detectData = await detectRes.json();
            if (detectData.error) throw new Error('检测失败: ' + detectData.error);
            addLog(`✅ 检测完成，共 ${detectData.total} 张图片，${detectData.total_count} 人`, 'success');

            // 重建
            addLog('步骤3：重建中...', 'info');
            const rebuildRes = await fetch('/api/rebuild', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    detect_result_folder: detectFolder.value.trim(),
                    mapping_file: dstFolder.value.trim() + '\\crop_mapping.xlsx',
                    original_img_folder: originalFolder.value.trim(),
                    output_folder: outputFolder.value.trim(),
                    nms_iou: parseFloat(nmsSlider.value)
                })
            });
            const rebuildData = await rebuildRes.json();
            if (!rebuildData.ok) throw new Error('重建失败: ' + rebuildData.msg);
            addLog(`✅ 重建完成！处理看台数: ${rebuildData.total_stands}，总人数: ${rebuildData.total_persons}`, 'success');
            addLog(`📁 结果保存至: ${rebuildData.output_folder}`, 'success');
            addLog('===== 工作流全部完成 =====', 'success');
        } catch (e) {
            addLog(`❌ 工作流中断: ${e.message}`, 'error');
        } finally {
            runAll.disabled = false;
        }
    });
})();