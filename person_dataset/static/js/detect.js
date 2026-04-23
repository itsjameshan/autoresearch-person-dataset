// static/js/detect.js
(function() {
    document.addEventListener('DOMContentLoaded', function() {
        // DOM 元素
        const folderPathInput = document.getElementById('folder-path-input');
        const startFolderDetectBtn = document.getElementById('start-folder-detect');
        const browseBtn = document.getElementById('browse-folder-btn');
        const folderSelectorInput = document.getElementById('folder-selector-input');
        const timeCostSpan = document.getElementById('time-cost');
        const targetCountSpan = document.getElementById('target-count');
        const totalPersonsSpan = document.getElementById('total-persons');
        const resultTbody = document.getElementById('result-tbody');
        const emptyTip = document.getElementById('empty-tip');
        const detectImg = document.getElementById('detect-img');
        const canvasOverlay = document.getElementById('canvas-overlay');
        const nextStepBtn = document.getElementById('next-step-btn');
        const gotoCropBtn = document.getElementById('gotoCrop');

        // 存储所有检测结果
        let allImageResults = [];
        let currentDisplayImage = null;
        let highlightedIndex = -1;

        // 从 URL 获取参数
        const urlParams = new URLSearchParams(window.location.search);
        const folderParam = urlParams.get('folder');
        const mappingParam = urlParams.get('mapping');
        const originalParam = urlParams.get('original');
        const timestampParam = urlParams.get('timestamp');
        const workRootParam = urlParams.get('workRoot');

        if (folderParam) {
            folderPathInput.value = folderParam;
            sessionStorage.setItem('detect_folder', folderParam);
        }
        if (mappingParam) sessionStorage.setItem('crop_mapping_file', mappingParam);
        if (originalParam) sessionStorage.setItem('crop_src_folder', originalParam);
        if (timestampParam) sessionStorage.setItem('crop_timestamp', timestampParam);
        if (workRootParam) sessionStorage.setItem('crop_work_root', workRootParam);

        // 浏览上传
        browseBtn.addEventListener('click', () => folderSelectorInput.click());
        folderSelectorInput.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            if (files.length === 0) return;
            const folderName = files[0].webkitRelativePath.split('/')[0];
            if (!folderName) return;

            emptyTip.textContent = `正在上传文件夹 ${folderName} ...`;
            browseBtn.disabled = true;
            startFolderDetectBtn.disabled = true;

            const formData = new FormData();
            files.forEach(file => formData.append('images', file, file.webkitRelativePath));

            try {
                const response = await fetch('/api/upload_folder_for_detect', { method: 'POST', body: formData });
                const result = await response.json();
                if (result.ok) {
                    folderPathInput.value = result.uploaded_path;
                    sessionStorage.setItem('detect_folder', result.uploaded_path);
                    if (result.work_root) sessionStorage.setItem('crop_work_root', result.work_root);
                    emptyTip.textContent = `文件夹已上传，请点击“开始检测”。`;
                } else {
                    alert('上传失败: ' + result.msg);
                }
            } catch (error) {
                alert('上传错误');
            } finally {
                browseBtn.disabled = false;
                startFolderDetectBtn.disabled = false;
                folderSelectorInput.value = '';
            }
        });

        // ========== 精确绘制函数（考虑 object-fit: contain 的偏移） ==========
        function getImageRenderRect(img) {
            const container = img.parentElement;
            const containerW = container.clientWidth;
            const containerH = container.clientHeight;
            const imgNaturalW = img.naturalWidth;
            const imgNaturalH = img.naturalHeight;
            if (imgNaturalW === 0 || imgNaturalH === 0) return null;

            const imgRatio = imgNaturalW / imgNaturalH;
            const containerRatio = containerW / containerH;

            let renderW, renderH, offsetX, offsetY;
            if (imgRatio > containerRatio) {
                renderW = containerW;
                renderH = containerW / imgRatio;
                offsetX = 0;
                offsetY = (containerH - renderH) / 2;
            } else {
                renderH = containerH;
                renderW = containerH * imgRatio;
                offsetX = (containerW - renderW) / 2;
                offsetY = 0;
            }
            return { renderW, renderH, offsetX, offsetY };
        }

        function drawAllBoxes(persons, highlightIdx = -1) {
            if (!detectImg.complete || detectImg.naturalWidth === 0) {
                setTimeout(() => drawAllBoxes(persons, highlightIdx), 50);
                return;
            }
            const canvas = canvasOverlay;
            const ctx = canvas.getContext('2d');
            const container = detectImg.parentElement;

            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const rect = getImageRenderRect(detectImg);
            if (!rect) return;

            const scaleX = rect.renderW / detectImg.naturalWidth;
            const scaleY = rect.renderH / detectImg.naturalHeight;

            // 绘制所有绿色框
            persons.forEach((p, idx) => {
                const x1 = rect.offsetX + p.x1 * scaleX;
                const y1 = rect.offsetY + p.y1 * scaleY;
                const w = (p.x2 - p.x1) * scaleX;
                const h = (p.y2 - p.y1) * scaleY;

                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, w, h);
            });

            // 绘制红色高亮框
            if (highlightIdx >= 0 && highlightIdx < persons.length) {
                const p = persons[highlightIdx];
                const x1 = rect.offsetX + p.x1 * scaleX;
                const y1 = rect.offsetY + p.y1 * scaleY;
                const w = (p.x2 - p.x1) * scaleX;
                const h = (p.y2 - p.y1) * scaleY;

                ctx.strokeStyle = '#ff3333';
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, w, h);

                ctx.fillStyle = "rgba(0,0,0,0.6)";
                ctx.fillRect(x1, y1 - 24, 140, 24);
                ctx.fillStyle = "#fff";
                ctx.font = "14px Arial";
                ctx.fillText(`person ${p.conf.toFixed(2)}`, x1 + 6, y1 - 6);
            }
        }

        function displayImageWithHighlight(imgRes, personIndex = -1) {
            if (!imgRes || !imgRes.image_data) return;
            detectImg.src = imgRes.image_data;
            detectImg.style.visibility = 'visible';
            currentDisplayImage = imgRes;
            highlightedIndex = personIndex;

            const persons = imgRes.persons || [];
            detectImg.onload = () => {
                drawAllBoxes(persons, personIndex);
                emptyTip.style.display = 'none';
            };
            if (detectImg.complete) {
                drawAllBoxes(persons, personIndex);
                emptyTip.style.display = 'none';
            }
        }

        // ========== 批量检测（添加计时逻辑） ==========
        async function performFolderDetection(folder) {
            emptyTip.textContent = '正在批量检测中...';
            startFolderDetectBtn.disabled = true;
            nextStepBtn.disabled = true;

            const startTime = performance.now();   // 开始计时

            try {
                const response = await fetch('/detect_folder', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: folder })
                });
                const data = await response.json();

                const endTime = performance.now();  // 结束计时
                const totalTime = (endTime - startTime) / 1000;
                timeCostSpan.textContent = totalTime.toFixed(2) + 's';

                if (data.error) {
                    alert('检测失败: ' + data.error);
                    emptyTip.textContent = '检测失败';
                    return;
                }

                emptyTip.textContent = `检测完成，共 ${data.total} 张图片，${data.total_count} 人`;
                allImageResults = data.results;
                updateResultTable(allImageResults);

                targetCountSpan.textContent = data.total_count + '个';
                totalPersonsSpan.textContent = data.total_count;

                const firstValid = allImageResults.find(r => r.persons && r.persons.length > 0);
                if (firstValid) {
                    displayImageWithHighlight(firstValid, -1);
                } else if (allImageResults.length > 0) {
                    displayImageWithHighlight(allImageResults[0], -1);
                }

                nextStepBtn.disabled = false;
            } catch (e) {
                alert('请求异常');
                timeCostSpan.textContent = '0.00s';
            } finally {
                startFolderDetectBtn.disabled = false;
            }
        }

        function updateResultTable(results) {
            resultTbody.innerHTML = '';
            let globalIndex = 1;
            results.forEach((imgRes, imgIdx) => {
                if (imgRes.persons && imgRes.persons.length > 0) {
                    imgRes.persons.forEach((p, personIdx) => {
                        const row = resultTbody.insertRow();
                        row.innerHTML = `
                            <td>${globalIndex}</td>
                            <td>${imgRes.name}</td>
                            <td>person</td>
                            <td>${p.conf.toFixed(2)}</td>
                            <td>(${p.x1},${p.y1})-(${p.x2},${p.y2})</td>
                        `;
                        row.dataset.imgIndex = imgIdx;
                        row.dataset.personIndex = personIdx;
                        row.addEventListener('click', () => {
                            document.querySelectorAll('#result-tbody tr').forEach(tr => tr.classList.remove('selected-row'));
                            row.classList.add('selected-row');
                            displayImageWithHighlight(results[imgIdx], personIdx);
                        });
                        globalIndex++;
                    });
                }
            });
            if (!document.getElementById('row-highlight-style')) {
                const style = document.createElement('style');
                style.id = 'row-highlight-style';
                style.textContent = `.selected-row { background-color: #e3f2fd !important; font-weight: bold; }`;
                document.head.appendChild(style);
            }
        }

        // 初始化提示
        const folderToDetect = folderPathInput.value.trim();
        emptyTip.textContent = folderToDetect ?
            `待检测文件夹：${folderToDetect}，请点击“开始检测”。` :
            '未检测到裁剪输出目录，请返回上一步或使用“浏览”上传。';

        startFolderDetectBtn.addEventListener('click', () => {
            const folder = folderPathInput.value.trim();
            if (!folder) { alert('请先完成裁剪或上传文件夹'); return; }
            performFolderDetection(folder);
        });

        gotoCropBtn.addEventListener('click', () => window.location.href = '/crop');
        nextStepBtn.addEventListener('click', () => {
            const folder = sessionStorage.getItem('detect_folder');
            const mapping = sessionStorage.getItem('crop_mapping_file');
            const original = sessionStorage.getItem('crop_src_folder');
            const timestamp = sessionStorage.getItem('crop_timestamp');
            const workRoot = sessionStorage.getItem('crop_work_root');
            if (!folder || !mapping) { alert('缺少必要参数'); return; }
            const url = `/rebuild?detect=${encodeURIComponent(folder)}&mapping=${encodeURIComponent(mapping)}&original=${encodeURIComponent(original)}&timestamp=${timestamp||''}&workRoot=${workRoot||''}`;
            window.location.href = url;
        });

        // 同步总人数
        const observer = new MutationObserver(() => {
            if (targetCountSpan && totalPersonsSpan) {
                const text = targetCountSpan.textContent.replace(/[^0-9]/g, '');
                if (text) totalPersonsSpan.textContent = text;
            }
        });
        observer.observe(targetCountSpan, { childList: true, characterData: true, subtree: true });
    });
})();