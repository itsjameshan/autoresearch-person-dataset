// static/js/login.js
(function() {
    // DOM 元素
    const tabLogin = document.getElementById('tabLogin');
    const tabRegister = document.getElementById('tabRegister');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const loginError = document.getElementById('loginError');
    const regError = document.getElementById('regError');
    const footerNote = document.getElementById('footerNote');

    // 输入框
    const loginUsername = document.getElementById('loginUsername');
    const loginPassword = document.getElementById('loginPassword');
    const regUsername = document.getElementById('regUsername');
    const regPassword = document.getElementById('regPassword');
    const regConfirm = document.getElementById('regConfirm');

    // 切换选项卡
    function switchTab(isLogin) {
        if (isLogin) {
            tabLogin.classList.add('active');
            tabRegister.classList.remove('active');
            loginForm.classList.add('active');
            registerForm.classList.remove('active');
            footerNote.style.display = 'block';
            clearErrors();
        } else {
            tabRegister.classList.add('active');
            tabLogin.classList.remove('active');
            registerForm.classList.add('active');
            loginForm.classList.remove('active');
            footerNote.style.display = 'none';
            clearErrors();
        }
    }

    function clearErrors() {
        loginError.textContent = '';
        regError.textContent = '';
    }

    tabLogin.addEventListener('click', () => switchTab(true));
    tabRegister.addEventListener('click', () => switchTab(false));

    // 登录提交
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = loginUsername.value.trim();
        const password = loginPassword.value;

        if (!username || !password) {
            loginError.textContent = '请输入用户名和密码';
            return;
        }

        const btn = loginForm.querySelector('button');
        btn.disabled = true;
        loginError.textContent = '';

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            if (response.ok && data.ok) {
                window.location.href = '/';
            } else {
                loginError.textContent = data.msg || '登录失败';
                btn.disabled = false;
            }
        } catch (error) {
            loginError.textContent = '网络错误，请稍后重试';
            btn.disabled = false;
        }
    });

    // 注册提交
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = regUsername.value.trim();
        const password = regPassword.value;
        const confirm = regConfirm.value;

        if (!username || !password || !confirm) {
            regError.textContent = '请填写所有字段';
            return;
        }
        if (username.length < 3) {
            regError.textContent = '用户名至少3位';
            return;
        }
        if (password.length < 6) {
            regError.textContent = '密码至少6位';
            return;
        }
        if (password !== confirm) {
            regError.textContent = '两次密码输入不一致';
            return;
        }

        const btn = registerForm.querySelector('button');
        btn.disabled = true;
        regError.textContent = '';

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            if (response.ok && data.ok) {
                alert('注册成功！请登录');
                switchTab(true);
                loginUsername.value = username;
                loginPassword.value = '';
                regUsername.value = '';
                regPassword.value = '';
                regConfirm.value = '';
            } else {
                regError.textContent = data.msg || '注册失败';
            }
        } catch (error) {
            regError.textContent = '网络错误，请稍后重试';
        } finally {
            btn.disabled = false;
        }
    });

    // 检查是否已登录（通过专用接口）
    async function checkAlreadyLogin() {
        try {
            const res = await fetch('/api/check_login');
            const data = await res.json();
            if (data.logged_in) {
                window.location.href = '/';
            }
        } catch (e) {
            // 忽略错误
        }
    }
    // 如需防止已登录用户重复登录，可取消下面一行注释
    //checkAlreadyLogin();

    // ===== 修复浏览器后退按钮导致登录按钮无法点击的问题 =====
    window.addEventListener('pageshow', (event) => {
        // 如果页面是从浏览器缓存（bfcache）恢复的
        if (event.persisted) {
            // 重置所有提交按钮为可用状态
            document.querySelectorAll('.auth-btn').forEach(btn => {
                btn.disabled = false;
            });
        }
    });
})();