(function syncPipelineBLink() {
    const a = document.getElementById('linkPipelineB');
    if (a && window.location && window.location.origin) {
        const url = window.location.origin + '/b';
        a.href = url;
        a.textContent = url;
    }
})();

async function runPipelineA() {
    const fileInput = document.getElementById('bigFile');
    const file = fileInput.files[0];
    const status = document.getElementById('aStatus');
    const btn = document.getElementById('btnRunA');

    if (!file) {
        alert('请先选择大图');
        return;
    }

    const overlap = document.getElementById('overlap').value || '200';
    const conf = document.getElementById('conf').value || '0.35';
    const iou = document.getElementById('iou').value || '0.25';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('overlap', overlap);
    formData.append('conf', conf);
    formData.append('iou', iou);

    status.textContent = '处理中（大图可能较慢）…';
    btn.disabled = true;

    try {
        const response = await fetch('/api/pipeline_a', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'pipeline_a 失败');
        }
        const text = JSON.stringify(data, null, 2);
        document.getElementById('aJson').value = text;
        status.textContent = `完成：${data.tiles?.length ?? 0} 块，sum_raw=${data.sum_raw ?? '?'}`;
        try {
            sessionStorage.setItem('pipeline_a_result', text);
        } catch (e) { /* ignore */ }
    } catch (e) {
        console.error(e);
        status.textContent = '';
        alert('失败：' + e.message);
    } finally {
        btn.disabled = false;
    }
}

function copyAtoClipboard() {
    const ta = document.getElementById('aJson');
    ta.select();
    document.execCommand('copy');
    alert('已复制到剪贴板');
}

function sendToPageB() {
    const text = document.getElementById('aJson').value.trim();
    if (!text) {
        alert('请先运行 pipeline_a');
        return;
    }
    try {
        sessionStorage.setItem('pipeline_a_result', text);
    } catch (e) { /* ignore */ }
    window.location.href = '/b';
}
