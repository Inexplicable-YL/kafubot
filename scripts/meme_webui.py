from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import imagehash
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from langchain_core.documents import Document

from agent.multimodal.image import (
    get_analyzer,
    read_image,
)
from agent.multimodal.meme import (
    MEME_CHROMA_PATH,
    MEME_PHASH_DISTANCE,
    add_memes,
    get_vectorstore,
)

FilterName = Literal["unlabeled", "labeled", "all"]
_CHROMA_LOCK = threading.RLock()
_LOCK_FILE_NAME = ".meme_label_webui.lock"
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_DUPLICATE_GROUP_MIN_SIZE = 2
_PENDING_NEW_ID_PREFIX = "pending-add:"
PRESET_ANALYSIS_PREFIXES = [
    "角色是花谱。",
    "角色是星界。",
    "角色是异世界情绪，又名情绪、情绪姐姐。",
]


@dataclass(frozen=True)
class MemeRecord:
    id: str
    analysis: str
    metadata: dict[str, Any]

    @property
    def manually_annotated(self) -> bool:
        value = self.metadata.get("manually_annotated", False)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)


@dataclass(frozen=True)
class DuplicateGroup:
    records: list[MemeRecord]
    min_distance: int
    max_distance: int
    pending_token: str | None = None


@dataclass(frozen=True)
class PendingDuplicateAdd:
    base64: str
    analysis: str
    phash: imagehash.ImageHash
    metadata: dict[str, str | bool]


_PENDING_DUPLICATE_ADDS: dict[str, PendingDuplicateAdd] = {}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meme 手动标注</title>
  <style>
    :root {
      --bg: #f6f1e7;
      --panel: #fffaf0;
      --ink: #27221b;
      --muted: #776f62;
      --line: #d8ccb8;
      --accent: #2f6b5f;
      --danger: #a33a2b;
      --shadow: rgba(70, 54, 28, 0.18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at 14% 16%, rgba(214, 162, 80, 0.24), transparent 28rem),
        radial-gradient(circle at 86% 12%, rgba(47, 107, 95, 0.18), transparent 24rem),
        linear-gradient(135deg, #f9f3e7 0%, var(--bg) 48%, #ede1cf 100%);
    }
    header {
      padding: 24px 28px 12px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: -0.04em;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 250, 240, 0.72);
    }
    main {
      padding: 12px 28px 28px;
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr);
      gap: 18px;
    }
    .panel {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 240, 0.86);
      box-shadow: 0 16px 42px var(--shadow);
      overflow: hidden;
    }
    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    select, button, textarea {
      font: inherit;
    }
    select {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px;
      background: #fffdf7;
      color: var(--ink);
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 9px 13px;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }
    button.secondary {
      color: var(--ink);
      background: #e7dac6;
    }
    button.danger {
      background: var(--danger);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.48;
    }
    .add-panel {
      margin: 0 28px 10px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 240, 0.78);
      box-shadow: 0 12px 30px var(--shadow);
    }
    .add-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr)) minmax(220px, 0.8fr);
      gap: 12px;
      align-items: end;
    }
    .field {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fffdf7;
      font: inherit;
    }
    .drop-zone {
      min-height: 76px;
      border: 1px dashed #9f8d70;
      border-radius: 16px;
      display: grid;
      place-items: center;
      padding: 12px;
      color: var(--muted);
      background: rgba(255, 253, 248, 0.72);
      cursor: pointer;
      text-align: center;
    }
    .drop-zone.dragover {
      border-color: var(--accent);
      color: var(--accent);
      background: rgba(47, 107, 95, 0.1);
    }
    .duplicate-panel {
      margin: 0 28px 10px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 240, 0.82);
      box-shadow: 0 12px 30px var(--shadow);
      overflow: hidden;
    }
    .duplicate-panel[hidden] {
      display: none;
    }
    .duplicate-body {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .duplicate-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .duplicate-card {
      display: grid;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
      background: #fffdf8;
    }
    .duplicate-card img {
      width: 100%;
      height: 180px;
      object-fit: contain;
      border-radius: 10px;
      box-shadow: none;
    }
    .duplicate-card .analysis-preview {
      min-height: 4.6em;
      max-height: 7em;
      overflow: auto;
      color: var(--ink);
      line-height: 1.45;
      font-size: 13px;
    }
    .keep-toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 14px;
    }
    .keep-toggle input {
      accent-color: var(--accent);
    }
    .image-wrap {
      padding: 18px;
      min-height: 440px;
      display: grid;
      place-items: center;
      background:
        linear-gradient(45deg, rgba(47, 107, 95, 0.08) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(47, 107, 95, 0.08) 25%, transparent 25%),
        #fffdf8;
      background-size: 24px 24px;
    }
    img {
      max-width: 100%;
      max-height: 70vh;
      object-fit: contain;
      border-radius: 16px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
      background: white;
    }
    .empty {
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }
    .editor {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    textarea {
      width: 100%;
      min-height: 360px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      color: var(--ink);
      background: #fffdf8;
      line-height: 1.55;
    }
    .meta {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
      font-size: 14px;
    }
    .preset-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .preset-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fffdf8;
      color: var(--ink);
      cursor: pointer;
      user-select: none;
      font-size: 14px;
    }
    .preset-tag input {
      accent-color: var(--accent);
    }
    @media (max-width: 860px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; padding: 8px 14px 18px; }
      .add-panel { margin: 0 14px 10px; }
      .duplicate-panel { margin: 0 14px 10px; }
      .add-grid { grid-template-columns: 1fr; }
      .image-wrap { min-height: 300px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Meme 手动标注</h1>
      <div class="stats">
        <span class="pill" id="count-all">全部: -</span>
        <span class="pill" id="count-unlabeled">未标注: -</span>
        <span class="pill" id="count-labeled">已标注: -</span>
      </div>
    </div>
    <div class="toolbar">
      <label>
        筛选
        <select id="filter">
          <option value="unlabeled">未人工标注</option>
          <option value="labeled">已人工标注</option>
          <option value="all">全部</option>
        </select>
      </label>
      <button class="secondary" id="prev">上一条</button>
      <button class="secondary" id="next">跳过/下一条</button>
      <button class="secondary" id="dedupe">全库查重</button>
    </div>
  </header>
  <section class="add-panel">
    <div class="add-grid">
      <label class="field">
        本地路径
        <span class="toolbar">
          <input id="add-path" type="text" placeholder="D:\path\to\meme.jpg">
          <button id="add-path-btn">分析添加</button>
        </span>
      </label>
      <label class="field">
        Web URL
        <span class="toolbar">
          <input id="add-url" type="text" placeholder="https://example.com/meme.jpg">
          <button id="add-url-btn">分析添加</button>
        </span>
      </label>
      <div class="field">
        图片拖动/上传
        <div class="drop-zone" id="drop-zone">拖入图片，或点击选择文件</div>
        <input id="add-file" type="file" accept="image/*" hidden>
      </div>
    </div>
  </section>
  <section class="duplicate-panel" id="duplicate-panel" hidden>
    <div class="panel-head">
      <strong>重复表情包审核</strong>
      <span class="pill" id="duplicate-position">-</span>
    </div>
    <div class="duplicate-body">
      <div class="toolbar">
        <button class="secondary" id="duplicate-prev">上一组</button>
        <button class="secondary" id="duplicate-next">下一组</button>
        <button id="duplicate-resolve">保留勾选并删除其余</button>
        <button class="secondary" id="duplicate-close">关闭</button>
      </div>
      <div class="status" id="duplicate-status"></div>
      <div class="duplicate-grid" id="duplicate-grid"></div>
    </div>
  </section>
  <main>
    <section class="panel">
      <div class="panel-head">
        <strong>图片</strong>
        <span class="pill" id="position">- / -</span>
      </div>
      <div class="image-wrap" id="image-wrap">
        <div class="empty">加载中...</div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <strong>Analysis</strong>
        <span class="pill" id="manual">-</span>
      </div>
      <div class="editor">
        <div class="preset-tags" id="preset-tags"></div>
        <textarea id="analysis" placeholder="当前 meme 没有 analysis"></textarea>
        <div class="meta">
          <div>ID: <span id="record-id">-</span></div>
          <div>URL: <span id="record-url">-</span></div>
        </div>
        <div class="actions">
          <div>
            <button id="save">保存并标为人工标注</button>
            <button class="danger" id="delete">删除 meme</button>
          </div>
          <div class="status" id="status"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const presetPrefixes = __PRESET_ANALYSIS_PREFIXES__;
    const state = {
      filter: "unlabeled",
      index: 0,
      record: null,
      total: 0,
      duplicateGroups: [],
      duplicateIndex: 0,
    };
    const el = (id) => document.getElementById(id);

    function setStatus(text, isError = false) {
      el("status").textContent = text;
      el("status").style.color = isError ? "var(--danger)" : "var(--muted)";
    }

    function setBusy(busy) {
      for (const id of [
        "prev", "next", "save", "delete", "filter", "dedupe",
        "duplicate-prev", "duplicate-next", "duplicate-resolve",
      ]) {
        el(id).disabled = busy;
      }
    }

    function setDuplicateStatus(text, isError = false) {
      el("duplicate-status").textContent = text;
      el("duplicate-status").style.color = isError ? "var(--danger)" : "var(--muted)";
    }

    function setAddBusy(busy) {
      for (const id of ["add-path", "add-url", "add-path-btn", "add-url-btn", "add-file"]) {
        el(id).disabled = busy;
      }
      el("drop-zone").style.pointerEvents = busy ? "none" : "";
      el("drop-zone").style.opacity = busy ? "0.55" : "";
    }

    function stripPresetPrefixes(text) {
      let body = text;
      let changed = true;
      while (changed) {
        changed = false;
        for (const prefix of presetPrefixes) {
          if (body.startsWith(prefix)) {
            body = body.slice(prefix.length);
            changed = true;
          }
        }
      }
      return body;
    }

    function selectedPresetPrefixes() {
      return Array.from(document.querySelectorAll(".preset-tag input:checked"))
        .map((input) => input.value);
    }

    function syncPresetChecks() {
      let current = el("analysis").value;
      for (const input of document.querySelectorAll(".preset-tag input")) {
        if (current.startsWith(input.value)) {
          input.checked = true;
          current = current.slice(input.value.length);
        } else {
          input.checked = false;
        }
      }
    }

    function applyPresetSelection() {
      const body = stripPresetPrefixes(el("analysis").value);
      el("analysis").value = selectedPresetPrefixes().join("") + body;
      el("analysis").focus();
    }

    function renderPresetTags() {
      const wrap = el("preset-tags");
      wrap.replaceChildren();
      for (const prefix of presetPrefixes) {
        const label = document.createElement("label");
        label.className = "preset-tag";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = prefix;
        input.addEventListener("change", applyPresetSelection);
        const text = document.createElement("span");
        text.textContent = prefix;
        label.appendChild(input);
        label.appendChild(text);
        wrap.appendChild(label);
      }
    }

    function renderEmpty(payload) {
      state.record = null;
      state.total = payload.total;
      const wrap = el("image-wrap");
      wrap.replaceChildren();
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "当前筛选没有可标注记录";
      wrap.appendChild(empty);
      el("analysis").value = "";
      syncPresetChecks();
      el("record-id").textContent = "-";
      el("record-url").textContent = "-";
      el("manual").textContent = "-";
      el("position").textContent = "0 / 0";
      updateCounts(payload.counts);
    }

    function updateCounts(counts) {
      el("count-all").textContent = `全部: ${counts.all}`;
      el("count-unlabeled").textContent = `未标注: ${counts.unlabeled}`;
      el("count-labeled").textContent = `已标注: ${counts.labeled}`;
    }

    function render(payload) {
      if (!payload.record) {
        renderEmpty(payload);
        return;
      }
      state.record = payload.record;
      state.index = payload.index;
      state.total = payload.total;
      updateCounts(payload.counts);
      el("position").textContent = `${payload.index + 1} / ${payload.total}`;
      el("record-id").textContent = payload.record.id;
      el("record-url").textContent = payload.record.url || "-";
      el("manual").textContent = payload.record.manually_annotated ? "已人工标注" : "未人工标注";
      el("analysis").value = payload.record.analysis || "";
      syncPresetChecks();
      const wrap = el("image-wrap");
      wrap.replaceChildren();
      if (payload.record.image_src) {
        const img = document.createElement("img");
        img.src = payload.record.image_src;
        img.alt = "meme";
        wrap.appendChild(img);
      } else {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "该记录没有 base64 图片";
        wrap.appendChild(empty);
      }
    }

    function renderDuplicateGroup() {
      el("duplicate-panel").hidden = false;
      const grid = el("duplicate-grid");
      grid.replaceChildren();

      if (!state.duplicateGroups.length) {
        el("duplicate-position").textContent = "0 / 0";
        setDuplicateStatus("没有发现需要审核的重复组");
        return;
      }

      state.duplicateIndex = Math.max(
        0,
        Math.min(state.duplicateIndex, state.duplicateGroups.length - 1),
      );
      const group = state.duplicateGroups[state.duplicateIndex];
      el("duplicate-position").textContent =
        `${state.duplicateIndex + 1} / ${state.duplicateGroups.length} · 距离 ${group.min_distance}-${group.max_distance}`;
      setDuplicateStatus("取消勾选要删除的记录；可保留一个或多个。");

      for (const record of group.records) {
        const card = document.createElement("article");
        card.className = "duplicate-card";
        if (record.image_src) {
          const img = document.createElement("img");
          img.src = record.image_src;
          img.alt = "meme";
          card.appendChild(img);
        }
        const keep = document.createElement("label");
        keep.className = "keep-toggle";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.value = record.id;
        keep.appendChild(checkbox);
        keep.appendChild(document.createTextNode("保留"));
        card.appendChild(keep);

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${record.pending ? "待新增" : "已有"} · ID: ${record.id} · phash: ${record.phash || "-"}`;
        card.appendChild(meta);

        const analysis = document.createElement("div");
        analysis.className = "analysis-preview";
        analysis.textContent = record.analysis || "";
        card.appendChild(analysis);
        grid.appendChild(card);
      }
    }

    async function loadDuplicates() {
      setBusy(true);
      setDuplicateStatus("查重中...");
      try {
        const resp = await fetch("/api/duplicates");
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "查重失败");
        state.duplicateGroups = payload.groups || [];
        state.duplicateIndex = 0;
        renderDuplicateGroup();
      } catch (err) {
        el("duplicate-panel").hidden = false;
        setDuplicateStatus(err.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function resolveDuplicateGroup() {
      if (!state.duplicateGroups.length) return;
      const group = state.duplicateGroups[state.duplicateIndex];
      const keepIds = Array.from(
        document.querySelectorAll("#duplicate-grid input[type='checkbox']:checked"),
      ).map((input) => input.value);
      if (!keepIds.length) {
        setDuplicateStatus("至少保留一条记录", true);
        return;
      }
      setBusy(true);
      setDuplicateStatus("处理中...");
      try {
        const resp = await fetch("/api/duplicates/resolve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            group_ids: group.records.map((record) => record.id),
            keep_ids: keepIds,
            pending_token: group.pending_token || null,
          }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "处理失败");
        await loadDuplicates();
        setDuplicateStatus(`已删除 ${payload.deleted} 条，保留 ${payload.kept} 条`);
      } catch (err) {
        setDuplicateStatus(err.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function loadRecord(index = state.index) {
      setBusy(true);
      setStatus("加载中...");
      try {
        const params = new URLSearchParams({ filter: state.filter, index: String(index) });
        const resp = await fetch(`/api/record?${params}`);
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "加载失败");
        render(payload);
        setStatus("");
      } catch (err) {
        setStatus(err.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function saveRecord() {
      if (!state.record) return;
      const analysis = el("analysis").value;
      if (!analysis.trim()) {
        setStatus("analysis 不能为空", true);
        return;
      }
      setBusy(true);
      setStatus("保存中...");
      try {
        const resp = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: state.record.id,
            analysis,
            original_analysis: state.record.analysis || "",
          }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "保存失败");
        setStatus(payload.reembedded ? "已保存并重新向量化" : "已标为人工标注");
        await loadRecord(state.filter === "unlabeled" ? state.index : state.index);
      } catch (err) {
        setStatus(err.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function deleteRecord() {
      if (!state.record) return;
      if (!confirm("确定删除这个 meme？此操作会从 Chroma 中移除该记录。")) return;
      setBusy(true);
      setStatus("删除中...");
      try {
        const resp = await fetch("/api/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: state.record.id,
            original_analysis: state.record.analysis || "",
          }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "删除失败");
        setStatus("已删除");
        await loadRecord(state.index);
      } catch (err) {
        setStatus(err.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function addFromJson(payload) {
      setAddBusy(true);
      setStatus("正在自动分析并添加...");
      try {
        const resp = await fetch("/api/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "添加失败");
        if (data.duplicate_review) {
          state.duplicateGroups = [data.group];
          state.duplicateIndex = 0;
          renderDuplicateGroup();
          setStatus("发现重复表情包，请在重复审核区选择保留项。");
          return;
        }
        state.filter = data.filter;
        el("filter").value = data.filter;
        render(data);
        setStatus("已自动分析，请检查 analysis 后保存人工标注");
      } catch (err) {
        setStatus(err.message, true);
      } finally {
        setAddBusy(false);
      }
    }

    async function addFromFile(file) {
      if (!file) return;
      setAddBusy(true);
      setStatus("正在上传、自动分析并添加...");
      try {
        const form = new FormData();
        form.append("file", file);
        const resp = await fetch("/api/add", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "添加失败");
        if (data.duplicate_review) {
          state.duplicateGroups = [data.group];
          state.duplicateIndex = 0;
          renderDuplicateGroup();
          setStatus("发现重复表情包，请在重复审核区选择保留项。");
          return;
        }
        state.filter = data.filter;
        el("filter").value = data.filter;
        render(data);
        setStatus("已自动分析，请检查 analysis 后保存人工标注");
      } catch (err) {
        setStatus(err.message, true);
      } finally {
        setAddBusy(false);
      }
    }

    el("filter").addEventListener("change", () => {
      state.filter = el("filter").value;
      state.index = 0;
      loadRecord(0);
    });
    el("prev").addEventListener("click", () => loadRecord(Math.max(0, state.index - 1)));
    el("next").addEventListener("click", () => loadRecord(state.index + 1));
    el("dedupe").addEventListener("click", loadDuplicates);
    el("save").addEventListener("click", saveRecord);
    el("delete").addEventListener("click", deleteRecord);
    el("duplicate-prev").addEventListener("click", () => {
      state.duplicateIndex -= 1;
      renderDuplicateGroup();
    });
    el("duplicate-next").addEventListener("click", () => {
      state.duplicateIndex += 1;
      renderDuplicateGroup();
    });
    el("duplicate-resolve").addEventListener("click", resolveDuplicateGroup);
    el("duplicate-close").addEventListener("click", () => {
      el("duplicate-panel").hidden = true;
    });
    el("add-path-btn").addEventListener("click", () => {
      const path = el("add-path").value.trim();
      if (!path) return setStatus("本地路径不能为空", true);
      addFromJson({ type: "path", value: path });
    });
    el("add-url-btn").addEventListener("click", () => {
      const url = el("add-url").value.trim();
      if (!url) return setStatus("URL 不能为空", true);
      addFromJson({ type: "url", value: url });
    });
    el("add-file").addEventListener("change", () => addFromFile(el("add-file").files[0]));
    el("drop-zone").addEventListener("click", () => el("add-file").click());
    el("drop-zone").addEventListener("dragover", (event) => {
      event.preventDefault();
      el("drop-zone").classList.add("dragover");
    });
    el("drop-zone").addEventListener("dragleave", () => {
      el("drop-zone").classList.remove("dragover");
    });
    el("drop-zone").addEventListener("drop", (event) => {
      event.preventDefault();
      el("drop-zone").classList.remove("dragover");
      addFromFile(event.dataTransfer.files[0]);
    });
    renderPresetTags();
    loadRecord(0);
  </script>
</body>
</html>
"""


def _collection():
    return get_vectorstore()._collection


@contextlib.contextmanager
def _chroma_operation_lock():
    lock_path = Path(MEME_CHROMA_PATH) / _LOCK_FILE_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with _CHROMA_LOCK, lock_path.open("r+b") as lock_file:
        if lock_path.stat().st_size == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)

        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _normalize_filter(value: str | None) -> FilterName:
    if value in {"unlabeled", "labeled", "all"}:
        return value  # type: ignore[return-value]
    return "unlabeled"


def _image_src(base64_value: Any) -> str:
    if not isinstance(base64_value, str) or not base64_value:
        return ""
    if base64_value.startswith("data:image/"):
        return base64_value
    if base64_value.startswith("base64://"):
        base64_value = base64_value.removeprefix("base64://")
    return f"data:image/jpeg;base64,{base64_value}"


def _read_records() -> list[MemeRecord]:
    with _chroma_operation_lock():
        return _read_records_unlocked()


def _read_records_unlocked() -> list[MemeRecord]:
    collection = _collection()
    total = collection.count()
    records: list[MemeRecord] = []
    batch_size = 200
    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for i, id_ in enumerate(ids):
            metadata = dict(metadatas[i] or {}) if i < len(metadatas) else {}
            analysis = documents[i] if i < len(documents) and documents[i] else ""
            records.append(MemeRecord(id=id_, analysis=analysis, metadata=metadata))
    return records


def _pending_record_id(token: str) -> str:
    return _PENDING_NEW_ID_PREFIX + token


def _records_similar_to_phash(
    records: list[MemeRecord],
    phash: imagehash.ImageHash,
) -> list[tuple[MemeRecord, int]]:
    matches: list[tuple[MemeRecord, int]] = []
    for record in records:
        existing_phash = _record_phash(record)
        if existing_phash is None:
            continue
        distance = int(existing_phash - phash)
        if distance < MEME_PHASH_DISTANCE:
            matches.append((record, distance))
    matches.sort(key=lambda item: (item[1], item[0].id))
    return matches


async def _analyze_and_add_source(
    source: str,
    *,
    metadata: dict[str, str | bool],
) -> dict[str, Any]:
    if source.startswith(("http://", "https://")):
        image = await read_image(path="", url=source)
    else:
        image = await read_image(path=source, url=None)
    if image is None:
        raise web.HTTPBadRequest(text="image analysis failed")

    analysis = await get_analyzer(use_cache=False).ainvoke(
        {
            "image": image.base64,
            "phash": image.phash,
        }
    )
    base64_value = "base64://" + image.base64

    with _chroma_operation_lock():
        records = _read_records_unlocked()
        similar_records = _records_similar_to_phash(records, image.phash)
        if similar_records:
            token = uuid.uuid4().hex
            _PENDING_DUPLICATE_ADDS[token] = PendingDuplicateAdd(
                base64=base64_value,
                analysis=analysis,
                phash=image.phash,
                metadata=metadata,
            )
            distances = [distance for _, distance in similar_records]
            group = DuplicateGroup(
                records=[record for record, _ in similar_records],
                min_distance=min(distances),
                max_distance=max(distances),
                pending_token=token,
            )
            return {
                "duplicate_review": True,
                "group": _serialize_duplicate_group(group),
            }

        added_ids = await add_memes([base64_value], [analysis], [image.phash])
        record_id = added_ids[0] if added_ids else None
        if record_id is None:
            raise web.HTTPConflict(text="similar meme already exists")

        record = _get_record_by_id(record_id)
        if record is None:
            raise web.HTTPInternalServerError(text="added record not readable")

        updated_metadata = dict(record.metadata)
        updated_metadata.update(metadata)
        updated_metadata.setdefault("manually_annotated", False)
        _collection().update(ids=[record_id], metadatas=[updated_metadata])
        records = _read_records_unlocked()
        counts = _counts(records)
        for index, added_record in enumerate(records):
            if added_record.id == record_id:
                return {
                    "filter": "all",
                    "index": index,
                    "total": len(records),
                    "counts": counts,
                    "record": _serialize_record(added_record),
                }
        raise web.HTTPInternalServerError(text="added record not readable")


def _validate_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise web.HTTPBadRequest(text="url must start with http:// or https://")
    return value


def _validate_local_path(value: str) -> str:
    path = Path(value)
    if not path.is_file():
        raise web.HTTPBadRequest(text="local path does not exist or is not a file")
    return str(path)


async def _save_upload_to_temp(request: web.Request) -> tuple[str, str]:
    reader = await request.multipart()
    field = await reader.next()
    while field is not None and (
        not isinstance(field, BodyPartReader) or field.name != "file"
    ):
        field = await reader.next()
    if not isinstance(field, BodyPartReader):
        raise web.HTTPBadRequest(text="file is required")

    filename = Path(field.filename or "uploaded-image").name
    suffix = Path(filename).suffix or ".img"
    fd, temp_path = tempfile.mkstemp(prefix="meme-label-upload-", suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as temp_file:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=_MAX_UPLOAD_BYTES,
                        actual_size=size,
                    )
                temp_file.write(chunk)
    except Exception:
        with contextlib.suppress(OSError):
            Path(temp_path).unlink()
        raise
    if size == 0:
        with contextlib.suppress(OSError):
            Path(temp_path).unlink()
        raise web.HTTPBadRequest(text="file is empty")
    return temp_path, filename


async def _add_from_json(request: web.Request) -> dict[str, Any]:
    payload = await _read_json(request)
    source_type = _required_string(payload, "type")
    value = _required_non_empty_text(payload, "value")
    if source_type == "url":
        source = _validate_url(value.strip())
        return await _analyze_and_add_source(
            source,
            metadata={"source_type": "url", "url": source},
        )
    if source_type == "path":
        source = _validate_local_path(value.strip())
        return await _analyze_and_add_source(
            source,
            metadata={"source_type": "path", "source_path": source},
        )
    raise web.HTTPBadRequest(text="type must be url or path")


async def _add_from_upload(request: web.Request) -> dict[str, Any]:
    temp_path, filename = await _save_upload_to_temp(request)
    try:
        return await _analyze_and_add_source(
            temp_path,
            metadata={"source_type": "upload", "filename": filename},
        )
    finally:
        with contextlib.suppress(OSError):
            Path(temp_path).unlink()


def _filter_records(
    records: list[MemeRecord], filter_name: FilterName
) -> list[MemeRecord]:
    if filter_name == "all":
        return records
    if filter_name == "labeled":
        return [record for record in records if record.manually_annotated]
    return [record for record in records if not record.manually_annotated]


def _counts(records: list[MemeRecord]) -> dict[str, int]:
    labeled = sum(1 for record in records if record.manually_annotated)
    return {
        "all": len(records),
        "labeled": labeled,
        "unlabeled": len(records) - labeled,
    }


def _record_phash(record: MemeRecord) -> imagehash.ImageHash | None:
    value = record.metadata.get("phash")
    if not isinstance(value, str) or not value:
        return None
    with contextlib.suppress(Exception):
        return imagehash.hex_to_hash(value)
    return None


def _dedupe_reviewed(record: MemeRecord) -> bool:
    value = record.metadata.get("dedupe_reviewed", False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _find_duplicate_groups(records: list[MemeRecord]) -> list[DuplicateGroup]:
    hashed: list[tuple[MemeRecord, imagehash.ImageHash]] = []
    for record in records:
        phash = _record_phash(record)
        if phash is not None:
            hashed.append((record, phash))
    parent = list(range(len(hashed)))
    distances: dict[tuple[int, int], int] = {}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, (_, left_phash) in enumerate(hashed):
        for j in range(i + 1, len(hashed)):
            distance = int(left_phash - hashed[j][1])
            if distance < MEME_PHASH_DISTANCE:
                distances[(i, j)] = distance
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for index in range(len(hashed)):
        grouped.setdefault(root(index), []).append(index)

    groups: list[DuplicateGroup] = []
    for indexes in grouped.values():
        if len(indexes) < _DUPLICATE_GROUP_MIN_SIZE:
            continue
        group_records = [hashed[index][0] for index in indexes]
        if all(_dedupe_reviewed(record) for record in group_records):
            continue
        index_set = set(indexes)
        group_distances = [
            distance
            for (left, right), distance in distances.items()
            if left in index_set and right in index_set
        ]
        groups.append(
            DuplicateGroup(
                records=group_records,
                min_distance=min(group_distances),
                max_distance=max(group_distances),
            )
        )

    groups.sort(key=lambda group: (group.min_distance, group.records[0].id))
    return groups


def _serialize_record(record: MemeRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "analysis": record.analysis,
        "manually_annotated": record.manually_annotated,
        "image_src": _image_src(record.metadata.get("base64")),
        "url": record.metadata.get("url", ""),
        "phash": record.metadata.get("phash", ""),
        "dedupe_reviewed": _dedupe_reviewed(record),
    }


def _serialize_duplicate_group(group: DuplicateGroup) -> dict[str, Any]:
    records = [_serialize_record(record) for record in group.records]
    if group.pending_token is not None:
        pending = _PENDING_DUPLICATE_ADDS[group.pending_token]
        records.insert(
            0,
            {
                "id": _pending_record_id(group.pending_token),
                "analysis": pending.analysis,
                "manually_annotated": False,
                "image_src": _image_src(pending.base64),
                "url": pending.metadata.get("url", ""),
                "phash": str(pending.phash),
                "dedupe_reviewed": False,
                "pending": True,
            },
        )
    return {
        "pending_token": group.pending_token,
        "min_distance": int(group.min_distance),
        "max_distance": int(group.max_distance),
        "records": records,
    }


def _get_record_by_id(id_: str) -> MemeRecord | None:
    result = _collection().get(ids=[id_], include=["documents", "metadatas"])
    ids = result.get("ids") or []
    if not ids:
        return None
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return MemeRecord(
        id=ids[0],
        analysis=documents[0] if documents and documents[0] else "",
        metadata=dict(metadatas[0] or {}) if metadatas else {},
    )


def _save_record(id_: str, analysis: str, original_analysis: str | None) -> bool:
    with _chroma_operation_lock():
        record = _get_record_by_id(id_)
        if record is None:
            raise web.HTTPNotFound(text="record not found")
        if original_analysis is not None and record.analysis != original_analysis:
            raise web.HTTPConflict(text="record changed, reload before saving")

        metadata = dict(record.metadata)
        metadata["manually_annotated"] = True

        if analysis != record.analysis:
            get_vectorstore().update_documents(
                ids=[id_],
                documents=[Document(page_content=analysis, metadata=metadata)],
            )
            return True

        _collection().update(ids=[id_], metadatas=[metadata])
        return False


def _delete_record(id_: str, original_analysis: str | None) -> None:
    with _chroma_operation_lock():
        record = _get_record_by_id(id_)
        if record is None:
            raise web.HTTPNotFound(text="record not found")
        if original_analysis is not None and record.analysis != original_analysis:
            raise web.HTTPConflict(text="record changed, reload before deleting")
        _collection().delete(ids=[id_])


def _resolve_duplicate_group(
    group_ids: list[str], keep_ids: list[str]
) -> dict[str, int]:
    keep_set = set(keep_ids)
    if not group_ids:
        raise web.HTTPBadRequest(text="group_ids is required")
    if not keep_set:
        raise web.HTTPBadRequest(text="keep_ids is required")
    if not keep_set.issubset(set(group_ids)):
        raise web.HTTPBadRequest(text="keep_ids must be part of group_ids")

    with _chroma_operation_lock():
        records = [record for id_ in group_ids if (record := _get_record_by_id(id_))]
        existing_ids = {record.id for record in records}
        if not keep_set.issubset(existing_ids):
            raise web.HTTPNotFound(text="kept record not found")

        kept_records = [record for record in records if record.id in keep_set]
        kept_metadatas = []
        for record in kept_records:
            metadata = dict(record.metadata)
            metadata["dedupe_reviewed"] = True
            kept_metadatas.append(metadata)
        if kept_records:
            _collection().update(
                ids=[record.id for record in kept_records],
                metadatas=kept_metadatas,
            )

        delete_ids = [record.id for record in records if record.id not in keep_set]
        if delete_ids:
            _collection().delete(ids=delete_ids)

    return {"kept": len(kept_records), "deleted": len(delete_ids)}


def _resolve_pending_duplicate_add(
    token: str,
    group_ids: list[str],
    keep_ids: list[str],
) -> dict[str, int | str | None]:
    pending_id = _pending_record_id(token)
    keep_set = set(keep_ids)
    if pending_id not in group_ids:
        raise web.HTTPBadRequest(text="pending record is not part of group_ids")
    if not keep_set:
        raise web.HTTPBadRequest(text="keep_ids is required")
    if not keep_set.issubset(set(group_ids)):
        raise web.HTTPBadRequest(text="keep_ids must be part of group_ids")

    with _chroma_operation_lock():
        pending = _PENDING_DUPLICATE_ADDS.get(token)
        if pending is None:
            raise web.HTTPNotFound(text="pending duplicate add not found")

        existing_group_ids = [id_ for id_ in group_ids if id_ != pending_id]
        records = [
            record
            for id_ in existing_group_ids
            if (record := _get_record_by_id(id_)) is not None
        ]
        existing_ids = {record.id for record in records}
        keep_existing_ids = keep_set - {pending_id}
        if not keep_existing_ids.issubset(existing_ids):
            raise web.HTTPNotFound(text="kept record not found")

        kept_records = [record for record in records if record.id in keep_existing_ids]
        kept_metadatas = []
        for record in kept_records:
            metadata = dict(record.metadata)
            metadata["dedupe_reviewed"] = True
            kept_metadatas.append(metadata)
        if kept_records:
            _collection().update(
                ids=[record.id for record in kept_records],
                metadatas=kept_metadatas,
            )

        delete_ids = [record.id for record in records if record.id not in keep_set]
        if delete_ids:
            _collection().delete(ids=delete_ids)

        new_id: str | None = None
        if pending_id in keep_set:
            metadata = dict(pending.metadata)
            metadata["base64"] = pending.base64
            metadata["phash"] = str(pending.phash)
            metadata["dedupe_reviewed"] = True
            metadata.setdefault("manually_annotated", False)
            ids = get_vectorstore().add_texts(
                texts=[pending.analysis],
                metadatas=[metadata],
            )
            new_id = ids[0] if ids else None

        del _PENDING_DUPLICATE_ADDS[token]

    return {
        "kept": len(keep_set),
        "deleted": len(delete_ids),
        "added": 1 if new_id is not None else 0,
        "new_id": new_id,
    }


def _parse_index(value: str | None) -> int:
    with contextlib.suppress(ValueError, TypeError):
        return max(0, int(value))  # type: ignore[arg-type]
    return 0


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text="invalid json") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="json object is required")
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise web.HTTPBadRequest(text=f"{key} is required")
    return value


def _required_non_empty_text(payload: dict[str, Any], key: str) -> str:
    value = _required_string(payload, key)
    if not value.strip():
        raise web.HTTPBadRequest(text=f"{key} is required")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise web.HTTPBadRequest(text=f"{key} must be a string")
    return value


def _required_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise web.HTTPBadRequest(text=f"{key} must be a string list")
    return value


async def index(_: web.Request) -> web.Response:
    html = HTML.replace(
        "__PRESET_ANALYSIS_PREFIXES__",
        json.dumps(PRESET_ANALYSIS_PREFIXES, ensure_ascii=False),
    )
    return web.Response(text=html, content_type="text/html")


async def get_record(request: web.Request) -> web.Response:
    filter_name = _normalize_filter(request.query.get("filter"))
    index = _parse_index(request.query.get("index"))

    try:
        records = await asyncio.to_thread(_read_records)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    filtered = _filter_records(records, filter_name)
    total = len(filtered)
    if total == 0:
        return web.json_response(
            {
                "filter": filter_name,
                "index": 0,
                "total": 0,
                "counts": _counts(records),
                "record": None,
            }
        )

    index = min(index, total - 1)
    return web.json_response(
        {
            "filter": filter_name,
            "index": index,
            "total": total,
            "counts": _counts(records),
            "record": _serialize_record(filtered[index]),
        }
    )


async def save_record(request: web.Request) -> web.Response:
    try:
        payload = await _read_json(request)
        id_ = _required_string(payload, "id")
        analysis = _required_non_empty_text(payload, "analysis")
        original_analysis = _optional_string(payload, "original_analysis")
        reembedded = await asyncio.to_thread(
            _save_record,
            id_,
            analysis,
            original_analysis,
        )
    except web.HTTPException as exc:
        return web.json_response({"error": exc.text}, status=exc.status)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, "reembedded": reembedded})


async def add_record(request: web.Request) -> web.Response:
    try:
        if request.content_type.startswith("multipart/"):
            response = await _add_from_upload(request)
        else:
            response = await _add_from_json(request)
    except web.HTTPException as exc:
        return web.json_response({"error": exc.text}, status=exc.status)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(response)


async def delete_record(request: web.Request) -> web.Response:
    try:
        payload = await _read_json(request)
        id_ = _required_string(payload, "id")
        original_analysis = _optional_string(payload, "original_analysis")
        await asyncio.to_thread(_delete_record, id_, original_analysis)
    except web.HTTPException as exc:
        return web.json_response({"error": exc.text}, status=exc.status)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True})


async def get_duplicates(_: web.Request) -> web.Response:
    try:
        records = await asyncio.to_thread(_read_records)
        groups = await asyncio.to_thread(_find_duplicate_groups, records)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(
        {
            "distance": int(MEME_PHASH_DISTANCE),
            "groups": [_serialize_duplicate_group(group) for group in groups],
        }
    )


async def resolve_duplicates(request: web.Request) -> web.Response:
    try:
        payload = await _read_json(request)
        group_ids = _required_string_list(payload, "group_ids")
        keep_ids = _required_string_list(payload, "keep_ids")
        pending_token = _optional_string(payload, "pending_token")
        if pending_token:
            result = await asyncio.to_thread(
                _resolve_pending_duplicate_add,
                pending_token,
                group_ids,
                keep_ids,
            )
        else:
            result = await asyncio.to_thread(
                _resolve_duplicate_group, group_ids, keep_ids
            )
    except web.HTTPException as exc:
        return web.json_response({"error": exc.text}, status=exc.status)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, **result})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/record", get_record)
    app.router.add_get("/api/duplicates", get_duplicates)
    app.router.add_post("/api/add", add_record)
    app.router.add_post("/api/save", save_record)
    app.router.add_post("/api/delete", delete_record)
    app.router.add_post("/api/duplicates/resolve", resolve_duplicates)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meme manual labeling WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
