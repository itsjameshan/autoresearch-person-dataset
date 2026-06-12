let lastCsvBase64 = null;

document.getElementById('bImage').addEventListener('change', function (e) {
    const f = e.target.files[0];
    document.getElementById('bImageName').textContent = f ? f.name : '';
});

window.addEventListener('DOMContentLoaded', function () {
    try {
        const cached = sessionStorage.getItem('pipeline_a_result');
        if (cached && !document.getElementById('bPayload').value.trim()) {
            document.getElementById('bPayload').value = cached;
        }
    } catch (e) { /* ignore */ }
});

async function runPipelineB() {
    const raw = document.getElementById('bPayload').value.trim();
    const status = document.getElementById('bStatus');
    const preview = document.getElementById('bPreview');

    if (!raw) {
        alert('请粘贴 pipeline_a 的 JSON');
        return;
    }

    try {
        JSON.parse(raw);
    } catch (e) {
        alert('JSON 格式无效');
        return;
    }

    const mergeIou = document.getElementById('mergeIou').value || '0.35';
    const imgFile = document.getElementById('bImage').files[0];

    status.textContent = '合并中…';
    preview.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('payload', raw);
        formData.append('merge_iou', mergeIou);
        if (imgFile) {
            formData.append('image', imgFile);
        }

        const response = await fetch('/api/pipeline_b', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'pipeline_b 失败');
        }

        document.getElementById('bCount').textContent = data.count;
        document.getElementById('bYolo').value = data.yolo_txt || '';
        lastCsvBase64 = data.csv_base64 || null;
        status.textContent = `完成：merge_iou=${data.merge_iou}，orig ${data.orig_w}×${data.orig_h}`;

        if (data.image_data) {
            const img = document.createElement('img');
            img.src = data.image_data;
            img.style.maxWidth = '100%';
            img.alt = 'merged preview';
            preview.appendChild(img);
        }
        if (data.image_error) {
            status.textContent += ' · ' + data.image_error;
        }
    } catch (e) {
        console.error(e);
        status.textContent = '';
        alert('失败：' + e.message);
    }
}

function downloadText(textareaId, filename) {
    const text = document.getElementById(textareaId).value;
    if (!text) {
        alert('无内容');
        return;
    }
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function downloadCsv() {
    if (!lastCsvBase64) {
        alert('请先成功运行 pipeline_b');
        return;
    }
    const binary = atob(lastCsvBase64);
    const arr = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        arr[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([arr], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'merge_summary_' + new Date().toISOString().slice(0, 19).replace(/:/g, '') + '.csv';
    a.click();
    URL.revokeObjectURL(url);
}
