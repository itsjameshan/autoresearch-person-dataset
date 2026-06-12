// static/js/index.js
(function() {
    'use strict';

    const navItems = document.querySelectorAll('.nav-item[data-page]');
    const welcomePage = document.getElementById('welcome-page');
    const mainContent = document.getElementById('main-content');
    const modelBadge = document.getElementById('current-model-display');

    // 切换右侧内容
    function switchPage(page) {
        // 更新导航高亮
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === page) item.classList.add('active');
        });

        // 处理各页面
        if (page === 'home') {
            welcomePage.style.display = 'flex';          // 显示欢迎页
            mainContent.style.display = 'none';          // 隐藏动态内容区
            mainContent.innerHTML = '';
        } else if (page === 'history') {
            welcomePage.style.display = 'none';
            mainContent.style.display = 'flex';          // ✅ 关键修复：设置为 flex
            mainContent.style.flexDirection = 'column';  // 纵向排列
            loadHistory();
        } else if (page === 'profile') {
            welcomePage.style.display = 'none';
            mainContent.style.display = 'flex';
            mainContent.style.flexDirection = 'column';
            loadProfile();
        }
    }

    // 点击导航按钮
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page === 'workflow') return; // 工作流是跳转，已用 onclick
            switchPage(page);
        });
    });

    // ========== 历史数据 ==========
    async function loadHistory() {
        mainContent.innerHTML = '<div style="text-align:center; padding:40px; color:#999;">加载中...</div>';
        try {
            const res = await fetch('/api/history/list');
            const data = await res.json();
            const projects = data.projects || [];
            if (projects.length === 0) {
                mainContent.innerHTML = '<h1>📋 历史检测记录</h1><p style="text-align:center; color:#999;">暂无历史记录</p>';
                return;
            }
            let html = `<h1>📋 历史检测记录</h1>
            <table>
                <thead><tr><th>项目名称</th><th>创建时间</th><th>总人数</th><th>看台数</th><th>操作</th></tr></thead>
                <tbody>`;
            projects.forEach(p => {
                html += `<tr>
                    <td>${p.name}</td>
                    <td>${p.create_time || '-'}</td>
                    <td>${p.person_count}</td>
                    <td>${p.stand_count}</td>
                    <td><button class="action-btn" style="padding:6px 16px; background:#2d6a4f; color:white; border:none; border-radius:14px;" onclick="downloadProject('${p.name}')">下载</button></td>
                </tr>`;
            });
            html += '</tbody></table>';
            mainContent.innerHTML = html;
        } catch (e) {
            mainContent.innerHTML = '<p style="text-align:center; color:#e74c3c;">加载失败</p>';
        }
    }

    window.downloadProject = async function(name) {
        try {
            const res = await fetch('/api/history/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project: name })
            });
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = name + '_结果.zip';
                a.click();
                URL.revokeObjectURL(url);
            } else {
                const err = await res.json();
                alert('下载失败: ' + err.msg);
            }
        } catch (e) {
            alert('下载出错');
        }
    };

    // ========== 个人中心 ==========
    async function loadProfile() {
        mainContent.innerHTML = '<div style="text-align:center; padding:40px; color:#999;">加载中...</div>';
        try {
            const res = await fetch('/api/user/profile');
            const data = await res.json();
            if (data.ok) {
                const initial = data.display_name.charAt(0).toUpperCase();
                mainContent.innerHTML = `
                    <div class="profile-section" style="display:flex; align-items:center; gap:20px; margin-bottom:30px;">
                        <div class="avatar-circle" style="width:72px;height:72px;border-radius:50%;background:linear-gradient(145deg, #1e6ec7, #4da8ff);color:white;font-size:32px;font-weight:700;display:flex;align-items:center;justify-content:center;box-shadow:0 8px 16px rgba(30,110,200,0.2);">${initial}</div>
                        <div class="profile-info">
                            <h2 style="font-size:26px;color:#1e3c5c;margin-bottom:6px;">${data.display_name}</h2>
                            <p style="font-size:14px;color:#5a7c9e;margin:4px 0;">身份：${data.role}</p>
                            <p style="font-size:14px;color:#5a7c9e;margin:4px 0;">用户名：${data.username}</p>
                        </div>
                    </div>
                    <div style="height:1px; background:linear-gradient(to right, #d0dde9, transparent); margin:20px 0;"></div>
                    <div class="action-section" style="display:flex; flex-direction:column; gap:12px;">
                        <button class="action-btn" id="changePasswordBtn" style="padding:14px 20px; border:1px solid #d0dde9; border-radius:12px; background:white; cursor:pointer; display:flex; align-items:center; gap:10px; font-size:15px;">
                            <span>🔒</span> 修改密码 <span style="margin-left:auto;">›</span>
                        </button>
                        <button class="action-btn logout-btn" id="logoutBtn" style="padding:14px 20px; border:1px solid #f0c0c0; border-radius:12px; background:#fff5f5; cursor:pointer; display:flex; align-items:center; gap:10px; font-size:15px; color:#c62828;">
                            <span>🚪</span> 退出登录 <span style="margin-left:auto;">›</span>
                        </button>
                    </div>
                `;
                // 绑定事件
                document.getElementById('changePasswordBtn').addEventListener('click', showPasswordModal);
                document.getElementById('logoutBtn').addEventListener('click', logout);
            } else {
                mainContent.innerHTML = '<p style="text-align:center; color:#e74c3c;">获取用户信息失败</p>';
            }
        } catch (e) {
            mainContent.innerHTML = '<p style="text-align:center; color:#e74c3c;">网络错误</p>';
        }
    }

    function showPasswordModal() {
        document.getElementById('oldPassword').value = '';
        document.getElementById('newPassword').value = '';
        document.getElementById('modalError').textContent = '';
        document.getElementById('passwordModal').classList.add('active');
    }

    document.getElementById('cancelBtn').addEventListener('click', () => {
        document.getElementById('passwordModal').classList.remove('active');
    });

    document.getElementById('saveBtn').addEventListener('click', async () => {
        const oldPwd = document.getElementById('oldPassword').value.trim();
        const newPwd = document.getElementById('newPassword').value.trim();
        if (!oldPwd || !newPwd) {
            document.getElementById('modalError').textContent = '请填写所有字段';
            return;
        }
        if (newPwd.length < 6) {
            document.getElementById('modalError').textContent = '新密码至少6位';
            return;
        }
        try {
            const res = await fetch('/api/user/change_password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_password: oldPwd, new_password: newPwd })
            });
            const data = await res.json();
            if (data.ok) {
                alert(data.msg);
                document.getElementById('passwordModal').classList.remove('active');
                await fetch('/api/logout', { method: 'POST' });
                window.location.href = '/login';
            } else {
                document.getElementById('modalError').textContent = data.msg;
            }
        } catch (e) {
            document.getElementById('modalError').textContent = '网络错误';
        }
    });

    async function logout() {
        if (confirm('确定要退出登录吗？')) {
            await fetch('/api/logout', { method: 'POST' });
            window.location.href = '/login';
        }
    }

    // 加载模型信息
    async function loadModelInfo() {
        try {
            const res = await fetch('/get_models');
            const data = await res.json();
            modelBadge.textContent = data.current || '未加载';
        } catch (e) {
            modelBadge.textContent = '获取失败';
        }
    }

    loadModelInfo();
    // 默认展示首页
    switchPage('home');
})();