// static/js/crop.js
(function() {
    // DOM 元素
    const modeRadios = document.querySelectorAll('input[name="workMode"]');
    const batchArea = document.getElementById('batchModeArea');
    const singleArea = document.getElementById('singleModeArea');

    // 批量模式元素
    const srcFolderDisplay = document.getElementById('srcFolderDisplay');
    const selectSrcLocalBtn = document.getElementById('selectSrcLocalBtn');
    const tileSizeInput = document.getElementById('tileSize');
    const overlapInput = document.getElementById('overlap');
    const filterEmptyCheck = document.getElementById('filterEmpty');
    const folderUploader = document.getElementById('folderUploader');

    // 单张模式元素
    const singleFileDisplay = document.getElementById('singleFileDisplay');
    const selectSingleFileBtn = document.getElementById('selectSingleFileBtn');
    const singleTileSizeInput = document.getElementById('singleTileSize');
    const singleOverlapInput = document.getElementById('singleOverlap');
    const singleFilterEmptyCheck = document.getElementById('singleFilterEmpty');
    const singleFileUploader = document.getElementById('singleFileUploader');

    const startCropBtn = document.getElementById('startCropBtn');
    const nextBtn = document.getElementById('nextBtn');
    const logBox = document.getElementById('logBox');

    let selectedFiles = [];        // 批量模式用
    let selectedSingleFile = null; // 单张模式用

    // 日志函数
    function addLog(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logBox.appendChild(entry);
        logBox.scrollTop = logBox.scrollHeight;
    }

    // 切换模式显示
    function updateModeUI() {
        const mode = document.querySelector('input[name="workMode"]:checked').value;
        if (mode === 'batch') {
            batchArea.classList.remove('hidden');
            singleArea.classList.add('hidden');
            startCropBtn.disabled = (selectedFiles.length === 0);
        } else {
            batchArea.classList.add('hidden');
            singleArea.classList.remove('hidden');
            startCropBtn.disabled = (selectedSingleFile === null);
        }
    }

    modeRadios.forEach(radio => radio.addEventListener('change', updateModeUI));

    // 批量模式：选择文件夹
    selectSrcLocalBtn.addEventListener('click', () => folderUploader.click());
    folderUploader.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        selectedFiles = files;
        const folderName = files[0].webkitRelativePath.split('/')[0];
        srcFolderDisplay.value = folderName;
        addLog(`已选择本地文件夹: ${folderName}，包含 ${files.length} 个文件`, 'info');
        if (document.querySelector('input[name="workMode"]:checked').value === 'batch') {
            startCropBtn.disabled = false;
        }
    });

    // 单张模式：选择图片文件
    selectSingleFileBtn.addEventListener('click', () => singleFileUploader.click());
    singleFileUploader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        selectedSingleFile = file;
        singleFileDisplay.value = file.name;
        addLog(`已选择单张图片: ${file.name}`, 'info');
        if (document.querySelector('input[name="workMode"]:checked').value === 'single') {
            startCropBtn.disabled = false;
        }
    });

    // 开始处理（批量裁剪或单张裁剪）
    startCropBtn.addEventListener('click', async () => {
        const mode = document.querySelector('input[name="workMode"]:checked').value;

        if (mode === 'batch' && selectedFiles.length === 0) {
            alert('请先选择原始大图文件夹');
            return;
        }
        if (mode === 'single' && !selectedSingleFile) {
            alert('请先选择一张图片');
            return;
        }

        startCropBtn.disabled = true;
        nextBtn.disabled = true;
        addLog(`开始${mode === 'batch' ? '批量裁剪' : '单张图片裁剪'}...`, 'info');

        const formData = new FormData();

        if (mode === 'batch') {
            // 批量模式：多个文件 + 批量参数
            for (let file of selectedFiles) {
                formData.append('images', file);
            }
            formData.append('tile_size', tileSizeInput.value);
            formData.append('overlap', overlapInput.value);
            formData.append('filter_empty', filterEmptyCheck.checked);
        } else {
            // 单张模式：一个文件 + 单张参数（与批量参数独立，但字段名相同）
            formData.append('images', selectedSingleFile);
            formData.append('tile_size', singleTileSizeInput.value);
            formData.append('overlap', singleOverlapInput.value);
            formData.append('filter_empty', singleFilterEmptyCheck.checked);
        }

        try {
            const response = await fetch('/api/upload_and_crop', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.ok) {
                addLog(`✅ 裁剪完成！共生成 ${result.total_tiles} 张小图`, 'success');
                addLog(`映射文件: ${result.mapping_file}`, 'success');

                sessionStorage.setItem('crop_dst_folder', result.dst_folder);
                sessionStorage.setItem('crop_mapping_file', result.mapping_file);
                sessionStorage.setItem('crop_src_folder', result.src_folder);
                sessionStorage.setItem('crop_timestamp', result.timestamp);
                sessionStorage.setItem('crop_work_root', result.work_root);

                nextBtn.disabled = false;
            } else {
                addLog(`❌ 裁剪失败: ${result.msg}`, 'error');
                startCropBtn.disabled = false;
            }
        } catch (error) {
            addLog(`❌ 请求错误: ${error.message}`, 'error');
            startCropBtn.disabled = false;
        }
    });

    // 下一步跳转
    nextBtn.addEventListener('click', () => {
        const dst = sessionStorage.getItem('crop_dst_folder');
        const mapping = sessionStorage.getItem('crop_mapping_file');
        const original = sessionStorage.getItem('crop_src_folder');
        const timestamp = sessionStorage.getItem('crop_timestamp');
        const workRoot = sessionStorage.getItem('crop_work_root');
        if (!dst || !mapping || !original) {
            alert('请先完成裁剪');
            return;
        }
        const url = `/detect?folder=${encodeURIComponent(dst)}&mapping=${encodeURIComponent(mapping)}&original=${encodeURIComponent(original)}&timestamp=${encodeURIComponent(timestamp || '')}&workRoot=${encodeURIComponent(workRoot || '')}`;
        window.location.href = url;
    });

    // 初始化
    startCropBtn.disabled = true;
    updateModeUI();
})();