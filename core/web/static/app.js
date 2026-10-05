(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    messages: $("#messages"),
    welcome: $("#welcome"),
    status: $("#status"),
    accountButton: $("#accountButton"),
    accountAvatar: $("#accountAvatar"),
    accountName: $("#accountName"),
    accountMenu: $("#accountMenu"),
    menuUsername: $("#menuUsername"),
    logoutButton: $("#logoutButton"),
    dialog: $("#authDialog"),
    authClose: $("#authClose"),
    authForm: $("#authForm"),
    username: $("#username"),
    password: $("#password"),
    authError: $("#authError"),
    authSubmit: $("#authSubmit"),
    composer: $("#composer"),
    text: $("#text"),
    files: $("#images"),
    previews: $("#previews"),
    send: $("#send"),
    toast: $("#toast")
  };

  const state = {
    socket: null,
    user: null,
    authMode: "login",
    pendingImages: [],
    liveBubbles: new Map(),
    reconnectTimer: null,
    reconnectAttempts: 0,
    manuallyClosed: false,
    historyKeys: new Set(),
    historyTimers: []
  };

  const basePath = location.pathname.replace(/\/$/, "");

  function getDeviceId() {
    let id = localStorage.getItem("botDeviceId");
    if (!id) {
      id = crypto.randomUUID
        ? crypto.randomUUID()
        : `d-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
      localStorage.setItem("botDeviceId", id);
    }
    return id;
  }

  function getSession() {
    return sessionStorage.getItem("botSession");
  }

  function setSession(token) {
    if (token) sessionStorage.setItem("botSession", token);
    else sessionStorage.removeItem("botSession");
  }

  function setConnectionStatus(text, status = "idle") {
    elements.status.textContent = text;
    elements.status.dataset.state = status;
  }

  function setUser(username) {
    state.user = username || null;
    const label = state.user || "登录";
    elements.accountName.textContent = label;
    elements.accountAvatar.textContent = state.user ? state.user.slice(0, 1).toUpperCase() : "访";
    elements.menuUsername.textContent = state.user || "未登录";
    elements.accountButton.setAttribute("aria-label", state.user ? `账号 ${state.user}` : "登录");
  }

  function showToast(message, duration = 2600) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { elements.toast.hidden = true; }, duration);
  }

  function hideWelcome() {
    if (elements.welcome) elements.welcome.hidden = true;
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      elements.messages.scrollTop = elements.messages.scrollHeight;
    });
  }

  function createMessageRow(role) {
    hideWelcome();
    const row = document.createElement("div");
    row.className = `message-row ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    row.append(bubble);
    elements.messages.append(row);
    scrollToBottom();
    return bubble;
  }

  function appendBubble(content, role, image = false) {
    const bubble = createMessageRow(role);
    if (image) {
      const img = document.createElement("img");
      img.src = content;
      img.alt = role === "user" ? "你发送的图片" : "助手发送的图片";
      bubble.append(img);
    } else {
      bubble.textContent = content;
    }
    scrollToBottom();
    return bubble;
  }

  function ensureLiveBubble(id) {
    const key = String(id);
    let bubble = state.liveBubbles.get(key);
    if (!bubble) {
      bubble = createMessageRow("bot");
      bubble.classList.add("streaming");
      state.liveBubbles.set(key, bubble);
    }
    return bubble;
  }

  function appendDelta(id, text) {
    const bubble = ensureLiveBubble(id);
    bubble.textContent += text;
    scrollToBottom();
  }

  function claimHistory(bubble, historyId) {
    if (!bubble || historyId === undefined || historyId === null) return;
    const key = `db:${historyId}`;
    state.historyKeys.add(key);
    bubble.closest(".message-row")?.setAttribute("data-history-id", key);
  }

  function endLive(id, historyId = null) {
    const key = String(id);
    const bubble = state.liveBubbles.get(key);
    if (bubble) {
      bubble.classList.remove("streaming");
      claimHistory(bubble, historyId);
      state.liveBubbles.delete(key);
    }
  }

  function renderHistory(messages) {
    if (!Array.isArray(messages)) return;
    for (const item of messages) {
      const key = `db:${item.id}`;
      if (state.historyKeys.has(key)) continue;
      const role = item.role === "user" ? "user" : (item.role === "notice" ? "notice" : "bot");
      const isImage = item.type === "image";
      const content = isImage ? String(item.data_url || "") : String(item.text || "");
      if (!content) continue;
      state.historyKeys.add(key);
      // 同一标签页重连时，本地气泡可能已经存在但尚无数据库 id；优先认领它，
      // 避免历史同步再画一遍。图片按 src 匹配，文本按正文匹配。
      const candidates = [...elements.messages.querySelectorAll(`.message-row.${role}:not([data-history-id]) .bubble`)];
      const existing = candidates.find((bubble) => isImage
        ? bubble.querySelector("img")?.src === content
        : bubble.textContent === content);
      if (existing) {
        existing.closest(".message-row")?.setAttribute("data-history-id", key);
        continue;
      }
      const bubble = appendBubble(content, role, isImage);
      bubble.closest(".message-row")?.setAttribute("data-history-id", key);
    }
  }

  function requestHistorySync() {
    if (state.socket?.readyState === WebSocket.OPEN && state.user) {
      state.socket.send(JSON.stringify({ type: "history_sync" }));
    }
  }

  function scheduleHistorySyncs() {
    state.historyTimers.forEach(clearTimeout);
    // 覆盖“重连时后台回复仍在生成”的竞态；服务端落库后会在下一次同步出现。
    state.historyTimers = [1500, 5000, 15000, 30000, 60000].map((delay) =>
      setTimeout(requestHistorySync, delay)
    );
  }

  async function api(path, body) {
    const response = await fetch(`${basePath}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* ignore invalid response body */ }
    return { response, data };
  }

  function openAuth(mode = "login") {
    setAuthMode(mode);
    elements.authError.textContent = "";
    elements.password.value = "";
    if (!elements.dialog.open) elements.dialog.showModal();
    setTimeout(() => elements.username.focus(), 30);
  }

  function closeAuth() {
    if (elements.dialog.open) elements.dialog.close();
  }

  function setAuthMode(mode) {
    state.authMode = mode;
    document.querySelectorAll(".tab").forEach((tab) => {
      const active = tab.dataset.tab === mode;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    const login = mode === "login";
    elements.authSubmit.textContent = login ? "登录" : "注册并登录";
    elements.password.autocomplete = login ? "current-password" : "new-password";
    elements.authError.textContent = "";
  }

  function scheduleReconnect() {
    if (!getSession() || state.manuallyClosed || state.reconnectTimer) return;
    const delay = Math.min(1000 * (2 ** state.reconnectAttempts), 12000);
    state.reconnectAttempts += 1;
    setConnectionStatus(`${Math.ceil(delay / 1000)} 秒后重连`, "busy");
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connect();
    }, delay);
  }

  function handleServerMessage(data) {
    switch (data.type) {
      case "ready":
        state.reconnectAttempts = 0;
        setUser(data.username);
        setConnectionStatus("在线", "online");
        closeAuth();
        scheduleHistorySyncs();
        break;
      case "history":
        renderHistory(data.messages);
        break;
      case "auth_error":
        setSession(null);
        setUser(null);
        state.manuallyClosed = true;
        setConnectionStatus("登录已失效", "error");
        openAuth("login");
        break;
      case "delta":
        appendDelta(data.id, data.text || "");
        break;
      case "delta_end":
        endLive(data.id, data.history_id);
        break;
      case "message":
        if (data.text) claimHistory(appendBubble(data.text, "bot"), data.history_id);
        break;
      case "image":
        if (data.data_url) claimHistory(appendBubble(data.data_url, "bot", true), data.history_id);
        break;
      case "notice":
        if (data.text) claimHistory(appendBubble(data.text, "notice"), data.history_id);
        break;
      default:
        break;
    }
  }

  function connect() {
    const token = getSession();
    if (!token) {
      setConnectionStatus("请先登录", "idle");
      openAuth("login");
      return;
    }
    if (state.socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.socket.readyState)) return;

    state.manuallyClosed = false;
    setConnectionStatus("正在连接", "busy");
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}${basePath}/ws`);
    state.socket = socket;

    socket.onopen = () => socket.send(JSON.stringify({ type: "auth", token }));
    socket.onmessage = (event) => {
      try { handleServerMessage(JSON.parse(event.data)); }
      catch (_) { showToast("收到无法解析的服务端消息"); }
    };
    socket.onerror = () => setConnectionStatus("连接异常", "error");
    socket.onclose = () => {
      if (state.socket === socket) state.socket = null;
      if (!state.manuallyClosed && getSession()) scheduleReconnect();
      else if (!getSession()) setConnectionStatus("请先登录", "idle");
    };
  }

  function disconnect() {
    state.manuallyClosed = true;
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    state.historyTimers.forEach(clearTimeout);
    state.historyTimers = [];
    if (state.socket) {
      try { state.socket.close(); } catch (_) { /* no-op */ }
      state.socket = null;
    }
  }

  async function submitAuth(event) {
    event.preventDefault();
    if (!elements.authForm.reportValidity()) return;

    elements.authError.textContent = "";
    elements.authSubmit.disabled = true;
    elements.authSubmit.textContent = state.authMode === "login" ? "登录中…" : "注册中…";
    try {
      const result = await api(state.authMode === "login" ? "/login" : "/register", {
        username: elements.username.value.trim(),
        password: elements.password.value,
        device_id: getDeviceId()
      });
      if (!result.response.ok || !result.data.ok) {
        elements.authError.textContent = result.data.error || "操作失败，请稍后重试";
        return;
      }
      setSession(result.data.token);
      setUser(result.data.username);
      closeAuth();
      connect();
    } catch (_) {
      elements.authError.textContent = "无法连接服务器，请检查网络";
    } finally {
      elements.authSubmit.disabled = false;
      elements.authSubmit.textContent = state.authMode === "login" ? "登录" : "注册并登录";
    }
  }

  function logout() {
    disconnect();
    setSession(null);
    setUser(null);
    elements.accountMenu.hidden = true;
    setConnectionStatus("已退出", "idle");
    openAuth("login");
  }

  function resizeTextarea() {
    elements.text.style.height = "auto";
    elements.text.style.height = `${Math.min(elements.text.scrollHeight, 160)}px`;
    updateSendState();
  }

  function updateSendState() {
    const connected = state.socket?.readyState === WebSocket.OPEN && state.user;
    const hasContent = elements.text.value.trim() || state.pendingImages.length;
    elements.send.disabled = !(connected && hasContent);
  }

  function renderPreviews() {
    elements.previews.replaceChildren();
    state.pendingImages.forEach((data, index) => {
      const item = document.createElement("div");
      item.className = "preview-item";
      const img = document.createElement("img");
      img.src = data;
      img.alt = `待发送图片 ${index + 1}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `移除图片 ${index + 1}`);
      remove.onclick = () => {
        state.pendingImages.splice(index, 1);
        renderPreviews();
        updateSendState();
      };
      item.append(img, remove);
      elements.previews.append(item);
    });
  }

  async function loadImages() {
    const selected = [...elements.files.files].slice(0, 4);
    if ([...elements.files.files].length > 4) showToast("每次最多选择 4 张图片");
    const loaded = [];
    for (const file of selected) {
      if (!file.type.startsWith("image/")) continue;
      const data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      loaded.push(data);
    }
    state.pendingImages = loaded;
    renderPreviews();
    updateSendState();
  }

  function sendMessage(event) {
    event.preventDefault();
    const value = elements.text.value.trim();
    if (!value && !state.pendingImages.length) return;
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN || !state.user) {
      showToast("尚未连接，正在尝试重连");
      connect();
      return;
    }

    const images = [...state.pendingImages];
    if (value) appendBubble(value, "user");
    images.forEach((image) => appendBubble(image, "user", true));
    state.socket.send(JSON.stringify({
      type: "message",
      id: String(Date.now()),
      text: value,
      images
    }));

    elements.text.value = "";
    elements.text.style.height = "auto";
    state.pendingImages = [];
    elements.files.value = "";
    renderPreviews();
    updateSendState();
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setAuthMode(tab.dataset.tab));
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.text.value = button.dataset.prompt;
      resizeTextarea();
      elements.text.focus();
    });
  });

  elements.authForm.addEventListener("submit", submitAuth);
  elements.authClose.addEventListener("click", () => {
    if (getSession()) closeAuth();
    else showToast("请先登录或注册");
  });
  elements.dialog.addEventListener("cancel", (event) => {
    if (!getSession()) event.preventDefault();
  });
  elements.accountButton.addEventListener("click", () => {
    if (!state.user) openAuth("login");
    else elements.accountMenu.hidden = !elements.accountMenu.hidden;
  });
  elements.logoutButton.addEventListener("click", logout);
  document.addEventListener("click", (event) => {
    if (!elements.accountMenu.hidden && !elements.accountMenu.contains(event.target) && !elements.accountButton.contains(event.target)) {
      elements.accountMenu.hidden = true;
    }
  });
  elements.files.addEventListener("change", loadImages);
  elements.text.addEventListener("input", resizeTextarea);
  elements.text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.composer.requestSubmit();
    }
  });
  elements.composer.addEventListener("submit", sendMessage);
  window.addEventListener("online", () => { showToast("网络已恢复"); connect(); });
  window.addEventListener("offline", () => setConnectionStatus("网络已断开", "error"));
  window.addEventListener("beforeunload", disconnect);

  setUser(null);
  updateSendState();
  if (getSession()) connect();
  else {
    setConnectionStatus("请先登录", "idle");
    openAuth("login");
  }
})();
