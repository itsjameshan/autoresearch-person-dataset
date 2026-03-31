// 单张检测
async function detectSingle() {
    const fileInput = document.getElementById('singleFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('请先选择图片');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/detect_single', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '检测失败');
        }

        const data = await response.json();

        // 显示检测结果图片（base64格式）
        document.getElementById('showImg').src = data.image_data;
        document.getElementById('total').innerText = data.count;

        // 渲染人物列表
        renderList(data.persons);

    } catch (error) {
        console.error('检测错误:', error);
        alert('检测失败：' + error.message);
    }
}

// 渲染人物列表
function renderList(persons) {
    const list = document.getElementById('personList');
    list.innerHTML = '';

    if (!persons || persons.length === 0) {
        list.innerHTML = '<div style="padding:20px;text-align:center;color:#999;">未检测到人物</div>';
        return;
    }

    persons.forEach((p, i) => {
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
            <strong>人物 ${i + 1}</strong><br>
            位置: (${p.x1}, ${p.y1}) - (${p.x2}, ${p.y2})<br>
            置信度: ${(p.conf * 100).toFixed(1)}%
        `;
        div.onclick = () => highlightBox(p);
        list.appendChild(div);
    });
}

// 高亮显示检测框
function highlightBox(box) {
    // 移除之前的高亮
    document.querySelectorAll('.box').forEach(b => b.remove());

    const img = document.getElementById('showImg');
    const canvas = document.getElementById('canvas');
    const canvasRect = canvas.getBoundingClientRect();

    if (!img.complete || img.naturalWidth === 0) {
        console.error('图片未加载完成');
        return;
    }

    // 计算缩放比例
    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;

    const boxDiv = document.createElement('div');
    boxDiv.className = 'box';

    // 计算相对位置
    const left = canvasRect.left + (box.x1 / scaleX);
    const top = canvasRect.top + (box.y1 / scaleY);
    const width = (box.x2 - box.x1) / scaleX;
    const height = (box.y2 - box.y1) / scaleY;

    boxDiv.style.left = left + 'px';
    boxDiv.style.top = top + 'px';
    boxDiv.style.width = width + 'px';
    boxDiv.style.height = height + 'px';

    document.body.appendChild(boxDiv);

    // 3秒后自动移除高亮
    setTimeout(() => {
        if (boxDiv.parentNode) boxDiv.remove();
    }, 3000);
}

// 批量检测
async function detectBatch() {
    const files = document.getElementById('batchFiles').files;

    if (!files.length) {
        alert('请先选择图片');
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }

    try {
        const response = await fetch('/detect_batch', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '批量检测失败');
        }

        const data = await response.json();

        // 显示批量结果
        let html = '<h4>检测结果：</h4>';
        data.results.forEach((r, idx) => {
            html += `
                <div class="batch-item">
                    <strong>${r.name}</strong><br>
                    人数：${r.count}
                    ${r.image_data ? `<button class="view-btn" onclick="viewBatchImage(${idx})">查看图片</button>` : ''}
                    ${r.error ? `<span style="color:red;margin-left:10px">错误: ${r.error}</span>` : ''}
                </div>
            `;
        });
        document.getElementById('batchResult').innerHTML = html;

        // 保存批量结果数据供查看使用
        window.batchResults = data.results;

        // 下载Excel文件
        if (data.excel_data) {
            downloadExcel(data.excel_data, data.excel_filename);
        }

        alert(`批量检测完成！共处理 ${data.results.length} 张图片`);

    } catch (error) {
        console.error('批量检测错误:', error);
        alert('批量检测失败：' + error.message);
    }
}

// 下载Excel文件
function downloadExcel(base64Data, filename) {
    const link = document.createElement('a');
    const binaryData = atob(base64Data);
    const array = new Uint8Array(binaryData.length);
    for (let i = 0; i < binaryData.length; i++) {
        array[i] = binaryData.charCodeAt(i);
    }
    const blob = new Blob([array], {type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// 查看批量检测的图片
function viewBatchImage(index) {
    if (window.batchResults && window.batchResults[index]) {
        const result = window.batchResults[index];
        if (result.image_data) {
            document.getElementById('showImg').src = result.image_data;
            document.getElementById('total').innerText = result.count;
            if (result.persons) {
                renderList(result.persons);
            }
        } else {
            alert('该图片无检测结果数据');
        }
    }
}

// 选择图片后自动显示预览
document.getElementById('singleFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('showImg').src = e.target.result;
            document.getElementById('total').innerText = '0';
            document.getElementById('personList').innerHTML = '';
        };
        reader.readAsDataURL(file);
    }
});

// 点击其他地方移除高亮
document.addEventListener('click', function(e) {
    if (!e.target.classList || !e.target.classList.contains('item')) {
        document.querySelectorAll('.box').forEach(b => b.remove());
    }
});