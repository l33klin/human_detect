const $ = (selector, root = document) => root.querySelector(selector);
let key = sessionStorage.getItem("hd-api-key") || "";
const cards = new Map();
const states = {
  connecting: "连接中",
  running: "运行中",
  reconnecting: "重新连接",
  busy: "识别繁忙",
  failed: "运行失败",
  stopped: "已停止",
};
const errors = {
  media_connection_failed: "输入或输出连接失败，正在重试。",
  media_timeout: "视频连接超时，正在重连。",
  media_processing_failed: "视频处理失败，正在重连。",
  tpu_timeout: "TPU 推理超时，需要重启服务。",
  tpu_unavailable: "TPU 不可用，请检查服务。",
  inference_busy: "识别队列繁忙，部分画面暂时不显示标注。",
  media_disabled: "服务尚未启用视频输出。",
  source_unavailable: "无法连接视频源。",
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (key) headers.set("Authorization", `Bearer ${key}`);
  const response = await fetch(path, {
    ...options,
    headers,
    cache: "no-store",
  });
  if (!response.ok) {
    if (response.status === 401)
      throw new Error("请在“连接设置”中填写有效的 API 密钥。");
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* WHEP can return plain text. */
    }
    throw new Error(
      typeof detail === "string" ? detail : `请求失败（${response.status}）`,
    );
  }
  return response;
}

function notice(message = "") {
  $("#notice").textContent = message;
  $("#notice").hidden = !message;
}
function openDialog(id) {
  const dialog = $(id);
  $(".form-error", dialog)?.replaceChildren();
  dialog.showModal();
}
$("#add-stream").onclick = $("#empty-add").onclick = () =>
  openDialog("#stream-dialog");
$("#add-identity").onclick = () => openDialog("#identity-dialog");
$("#access-button").onclick = () => {
  $("#access-form").elements.key.value = key;
  openDialog("#access-dialog");
};
document
  .querySelectorAll(".close-dialog")
  .forEach(
    (button) => (button.onclick = () => button.closest("dialog").close()),
  );
document.querySelectorAll(".tab").forEach(
  (button) =>
    (button.onclick = () => {
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab === button);
        tab.setAttribute("aria-pressed", String(tab === button));
      });
      $("#streams-view").hidden = button.dataset.view !== "streams";
      $("#identities-view").hidden = button.dataset.view !== "identities";
    }),
);

async function submit(form, action) {
  const button = $(".submit", form);
  button.disabled = true;
  try {
    await action(new FormData(form));
    form.closest("dialog").close();
    form.reset();
    await refresh();
  } catch (error) {
    $(".form-error", form).textContent = error.message;
  } finally {
    button.disabled = false;
  }
}
$("#stream-form").onsubmit = (event) => {
  event.preventDefault();
  submit(event.currentTarget, async (data) => {
    const source = data.get("source").trim();
    if (!/^rtsps?:\/\//i.test(source))
      throw new Error("请输入 rtsp:// 或 rtsps:// 地址。");
    await api("/v1/streams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, name: data.get("name"), publish: true }),
    });
  });
};
$("#identity-form").onsubmit = (event) => {
  event.preventDefault();
  submit(event.currentTarget, async (data) => {
    const query = new URLSearchParams({ name: data.get("name") });
    await api(
      `/v1/identities/${encodeURIComponent(data.get("person_id"))}/samples?${query}`,
      { method: "POST", body: data.get("sample") },
    );
  });
};
$("#access-form").onsubmit = async (event) => {
  event.preventDefault();
  key = event.currentTarget.elements.key.value.trim();
  if (key) sessionStorage.setItem("hd-api-key", key);
  else sessionStorage.removeItem("hd-api-key");
  for (const card of cards.values()) card.viewer.reset();
  $("#access-dialog").close();
  await refresh();
};

function iceComplete(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = () => {
      if (pc.iceGatheringState === "complete") {
        clearTimeout(timer);
        pc.removeEventListener("icegatheringstatechange", finish);
        resolve();
      }
    };
    const timer = setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", finish);
      reject(new Error("无法建立视频连接"));
    }, 6000);
    pc.addEventListener("icegatheringstatechange", finish);
  });
}

class Viewer {
  constructor(root) {
    this.root = root;
    this.video = $("video", root);
    this.message = $(".player-message", root);
    this.enabled = false;
    this.generation = 0;
    this.retry = 1000;
    this.video.onplaying = () => {
      this.message.hidden = true;
    };
  }
  async release() {
    this.pc?.close();
    this.pc = null;
    this.video.srcObject = null;
    const session = this.session;
    this.session = null;
    if (session) {
      try {
        await api(session, { method: "DELETE" });
      } catch {
        /* Expired sessions are harmless. */
      }
    }
  }
  reset() {
    this.generation++;
    this.enabled = false;
    clearTimeout(this.timer);
    this.release();
  }
  update(stream) {
    this.url = stream.output?.whep_url;
    const active = !!this.url && ["running", "busy"].includes(stream.state);
    if (!active) {
      if (this.enabled) this.reset();
      this.message.textContent =
        stream.state === "stopped"
          ? "视频流已停止"
          : stream.state === "failed"
            ? "视频暂不可用，请检查服务状态"
            : "正在等待视频…";
      this.message.hidden = false;
    } else if (!this.enabled) {
      this.enabled = true;
      this.connect(this.generation);
    }
  }
  async connect(generation) {
    if (!this.enabled || generation !== this.generation) return;
    this.message.hidden = false;
    this.message.textContent = "正在连接实时画面…";
    let pc, session;
    try {
      if (!window.RTCPeerConnection) throw new Error("此浏览器不支持实时视频");
      pc = new RTCPeerConnection({ iceServers: [] });
      this.pc = pc;
      const remote = new MediaStream();
      this.video.srcObject = remote;
      pc.ontrack = (event) => {
        remote.addTrack(event.track);
        this.video.play().catch(() => {
          this.message.hidden = false;
          this.message.textContent = "请点击播放器中的播放按钮观看";
        });
      };
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });
      await pc.setLocalDescription(await pc.createOffer());
      await iceComplete(pc);
      if (!this.enabled || generation !== this.generation) {
        pc.close();
        return;
      }
      const response = await api(this.url, {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: pc.localDescription.sdp,
      });
      session = response.headers.get("Location");
      if (!this.enabled || generation !== this.generation) {
        pc.close();
        if (session) api(session, { method: "DELETE" }).catch(() => {});
        return;
      }
      this.session = session;
      await pc.setRemoteDescription({
        type: "answer",
        sdp: await response.text(),
      });
      pc.onconnectionstatechange = () => {
        if (generation !== this.generation) return;
        if (pc.connectionState === "connected") {
          this.retry = 1000;
          this.message.hidden = true;
        } else if (["failed", "disconnected"].includes(pc.connectionState))
          this.reconnect(generation);
      };
      this.timer = setTimeout(() => {
        if (pc.connectionState !== "connected") this.reconnect(generation);
      }, 12000);
    } catch (error) {
      if (generation !== this.generation) {
        pc?.close();
        return;
      }
      this.message.textContent = `${error.message}，稍后重试…`;
      this.reconnect(generation);
    }
  }
  reconnect(generation) {
    if (!this.enabled || generation !== this.generation) return;
    clearTimeout(this.timer);
    this.generation++;
    const next = this.generation;
    this.message.hidden = false;
    this.message.textContent = "视频连接中断，正在重新连接…";
    this.release();
    this.timer = setTimeout(() => this.connect(next), this.retry);
    this.retry = Math.min(8000, this.retry * 2);
  }
}

function streamCard(stream) {
  const root = $("#stream-template").content.firstElementChild.cloneNode(true);
  const card = { root, viewer: new Viewer(root), stream };
  $(".copy", root).onclick = async (event) => {
    const input = $(".output-url", root);
    input.select();
    try {
      if (navigator.clipboard) await navigator.clipboard.writeText(input.value);
      else if (!document.execCommand("copy")) throw new Error();
      event.target.textContent = "已复制";
    } catch {
      event.target.textContent = "请手动复制";
    }
    setTimeout(() => {
      event.target.textContent = "复制地址";
    }, 1800);
  };
  for (const [selector, action] of [
    [".toggle", "toggle"],
    [".remove", "delete"],
  ]) {
    $(selector, root).onclick = async (event) => {
      if (action === "delete" && !confirm("删除此视频流及其连接配置？")) return;
      event.target.disabled = true;
      try {
        await api(
          `/v1/streams/${stream.stream_id}${action === "toggle" ? "/" + (card.stream.state === "stopped" || card.stream.state === "failed" ? "start" : "stop") : ""}`,
          { method: action === "delete" ? "DELETE" : "POST" },
        );
        await refresh();
      } catch (error) {
        notice(error.message);
      } finally {
        event.target.disabled = false;
      }
    };
  }
  $("#stream-grid").append(root);
  return card;
}

function renderStreams(streams) {
  const present = new Set(streams.map((s) => s.stream_id));
  for (const [id, card] of cards)
    if (!present.has(id)) {
      card.viewer.reset();
      card.root.remove();
      cards.delete(id);
    }
  $("#streams-empty").hidden = streams.length > 0;
  for (const stream of streams) {
    if (!cards.has(stream.stream_id))
      cards.set(stream.stream_id, streamCard(stream));
    const card = cards.get(stream.stream_id),
      root = card.root;
    card.stream = stream;
    $("h2", root).textContent =
      stream.name || `摄像头 ${stream.stream_id.slice(0, 6)}`;
    const status = $(".status", root);
    status.textContent = states[stream.state] || stream.state;
    status.dataset.state = stream.state;
    const media = stream.media;
    $(".format", root).textContent = media?.width
      ? `${media.width} × ${media.height} · 输出 ${(media.output_fps || 0).toFixed(1)} FPS · ${media.audio ? "有音频" : "无音频"}`
      : "等待视频信息";
    $(".targets", root).textContent = stream.latest
      ? `${["running", "busy"].includes(stream.state) ? "" : "上次 "}${stream.latest.persons.length} 人`
      : "";
    $(".stream-error", root).textContent =
      errors[stream.error] ||
      (stream.error ? "处理异常，请检查服务状态。" : "");
    $(".output-url", root).value =
      stream.output?.rtsp_url || "此视频流仅提供 JSON 结果";
    $(".copy", root).disabled = !stream.output?.rtsp_url;
    $(".toggle", root).textContent = ["stopped", "failed"].includes(
      stream.state,
    )
      ? "启动"
      : "停止";
    card.viewer.update(stream);
  }
}

function renderPeople(people) {
  const list = $("#identity-list");
  list.replaceChildren();
  $("#identities-empty").hidden = people.length > 0;
  for (const person of people) {
    const row = document.createElement("tr");
    for (const value of [person.name, person.person_id, person.samples]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const cell = document.createElement("td"),
      button = document.createElement("button");
    button.className = "danger";
    button.textContent = "删除";
    button.onclick = async () => {
      if (!confirm(`删除“${person.name}”及其全部人脸特征？`)) return;
      button.disabled = true;
      try {
        await api(`/v1/identities/${encodeURIComponent(person.person_id)}`, {
          method: "DELETE",
        });
        await refresh();
      } catch (error) {
        notice(error.message);
        button.disabled = false;
      }
    };
    cell.append(button);
    row.append(cell);
    list.append(row);
  }
}

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const [streams, people, config] = await Promise.all(
      ["/v1/streams", "/v1/identities", "/v1/config"].map(async (path) =>
        (await api(path)).json(),
      ),
    );
    renderStreams(streams);
    renderPeople(people);
    notice(
      config.media_enabled
        ? ""
        : "视频输出尚未启用。请按项目文档启动 MediaMTX，并设置 HD_MEDIA_ENABLED=true。现有图片和 JSON 接口仍可使用。",
    );
    $("#add-stream").disabled = $("#empty-add").disabled =
      !config.media_enabled;
  } catch (error) {
    notice(error.message);
  } finally {
    refreshing = false;
  }
}
window.addEventListener("pagehide", () => {
  for (const card of cards.values()) card.viewer.reset();
});
refresh();
setInterval(refresh, 2000);
