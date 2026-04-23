// static/js/index.js
(function() {
    'use strict';

    // 导航按钮跳转
    document.getElementById('nav-home').addEventListener('click', () => {
        window.location.href = '/';
    });

    document.getElementById('nav-workflow').addEventListener('click', () => {
        window.location.href = '/crop';
    });

    document.getElementById('nav-history').addEventListener('click', () => {
        alert('历史数据功能开发中，敬请期待');
    });

    document.getElementById('nav-profile').addEventListener('click', () => {
        alert('个人中心功能开发中，敬请期待');
    });

    // 获取当前模型名称
    async function loadModelInfo() {
        try {
            const res = await fetch('/get_models');
            const data = await res.json();
            const modelDisplay = document.getElementById('current-model-display');
            if (modelDisplay) {
                modelDisplay.textContent = data.current || '未加载';
            }
        } catch (e) {
            const modelDisplay = document.getElementById('current-model-display');
            if (modelDisplay) {
                modelDisplay.textContent = '获取失败';
            }
        }
    }

    loadModelInfo();
})();