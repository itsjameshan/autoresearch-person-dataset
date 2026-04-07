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

function syncOverlayCanvasSize() {
    const img = document.getElementById('detect-img');
    const canvas = document.getElementById('canvas-overlay');
    if (!img || !canvas || img.offsetWidth < 1) return;
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
}

/** 原图像素 (px,py) → 覆盖 canvas 坐标（含 object-fit: contain 居中留白） */
function imgPxToCanvas(px, py, img) {
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    const ew = img.offsetWidth;
    const eh = img.offsetHeight;
    if (!nw || !nh) return [0, 0];
    const s = Math.min(ew / nw, eh / nh);
    const dw = nw * s;
    const dh = nh * s;
    const ox = (ew - dw) / 2;
    const oy = (eh - dh) / 2;
    return [ox + px * s, oy + py * s];
}

document.getElementById('detect-img').addEventListener('load', () => {
    syncOverlayCanvasSize();
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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

// ====================== ✅ 开始检测：进度条 100% 必现 ✅ ======================
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

    // 🔥🔥🔥 强制把进度条直接插到页面正中间，不管样式是什么 🔥🔥🔥
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
                        <td>${p.class_name}</td>
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
                document.getElementById('detect-img').onload = function() {
                    canvas.width = this.offsetWidth;
                    canvas.height = this.offsetHeight;
                };
            }
        }

    } catch (err) {
        progressBar.remove();
        alert('检测失败');
        console.error(err);
    }
});

// ========= 下面代码完全没动，保持你原来的 =========
function renderSingleResult(data, file) {
    const tbody = document.getElementById('result-tbody');
    tbody.innerHTML = '';

    const persons = data.persons || [];
    persons.forEach((p, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i+1}</td>
            <td>${file.name}</td>
            <td>${p.class_name}</td>
            <td>${p.conf.toFixed(2)}</td>
            <td>${p.x1},${p.y1},${p.x2},${p.y2}</td>
        `;
        tr.onclick = () => highlightPerson(i);
        tbody.appendChild(tr);
    });

    document.getElementById('coord-x1').value = 0;
    document.getElementById('coord-y1').value = 0;
    document.getElementById('coord-x2').value = 0;
    document.getElementById('coord-y2').value = 0;

    if(document.getElementById('conf-input')){
        document.getElementById('conf-input').value = '';
    }

    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('detect-img');
    syncOverlayCanvasSize();
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

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

    if(document.getElementById('conf-input')){
        document.getElementById('conf-input').value = p.conf.toFixed(2);
    }

    const img = document.getElementById('detect-img');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');

    syncOverlayCanvasSize();
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [cx1, cy1] = imgPxToCanvas(p.x1, p.y1, img);
    const [cx2, cy2] = imgPxToCanvas(p.x2, p.y2, img);
    const x1 = Math.min(cx1, cx2);
    const y1 = Math.min(cy1, cy2);
    const w = Math.abs(cx2 - cx1);
    const h = Math.abs(cy2 - cy1);

    ctx.beginPath();
    ctx.rect(x1, y1, w, h);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#ff3333";
    ctx.stroke();

    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x1, y1 - 24, 130, 24);
    ctx.fillStyle = "#fff";
    ctx.font = "14px Arial";
    ctx.fillText(`person ${p.conf.toFixed(2)}`, x1 + 6, y1 - 6);
}

function removeProgressBar(){}