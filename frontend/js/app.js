/* 股票预测系统 - 前端逻辑（阶段A：内置模拟接口） */
(function () {
  "use strict";

  var state = { busy: false };

  // ---------- DOM ----------
  var $code = document.getElementById("stock-code");
  var $btn = document.getElementById("btn-predict");
  var $retrain = document.getElementById("btn-retrain");
  var $horizon = document.getElementById("horizon");
  var $horizonValue = document.getElementById("horizon-value");
  var $horizonWarning = document.getElementById("horizon-warning");
  var $progressArea = document.getElementById("progress-area");
  var $progressText = document.getElementById("progress-text");
  var $progressFill = document.getElementById("progress-fill");
  var $progressSpinner = document.getElementById("progress-spinner");
  var $statusLine = document.getElementById("status-line");
  var $resultArea = document.getElementById("result-area");
  var $errorBanner = document.getElementById("error-banner");
  var $metricsBody = document.querySelector("#metrics-table tbody");
  var $dataInfo = document.getElementById("data-info");
  var $forecastNote = document.getElementById("forecast-note");
  var $footerSource = document.getElementById("footer-source");
  var $historyDropdown = document.getElementById("history-dropdown");
  var $btnAdvanced = document.getElementById("btn-advanced");
  var $advancedPanel = document.getElementById("advanced-panel");
  var $advancedGeneral = document.getElementById("advanced-general");
  var $advancedModels = document.getElementById("advanced-models");
  var $btnResetAdvanced = document.getElementById("btn-reset-advanced");
  var $modalOverlay = document.getElementById("modal-overlay");
  var $modalMessage = document.getElementById("modal-message");
  var $modalConfirm = document.getElementById("modal-confirm");
  var $modalCancel = document.getElementById("modal-cancel");

  var chartTest = null, chartForecast = null;
  var highlightBest = -1;
  var modalCallback = null;

  // ---------- 常量 ----------
  var HISTORY_KEY = "prediction_history_v1";
  var ADVANCED_KEY = "advanced_options_v1";

  var ADVANCED_DEFAULTS = {
    max_rows: 1500,
    train_ratio: 0.8,
    models: ["ARIMA", "LSTM", "SVR", "BP", "ARIMA-LSTM"],
    garch_seed: 42,
    params: {
      "ARIMA": { max_p: 10, max_q: 10 },
      "LSTM": { time_step: 5, units: 48, dropout: 0.1, epochs: 50, batch: 32 },
      "SVR": { C: 100, epsilon: 0.1, gamma: "scale" },
      "BP": { time_step: 30, units_1: 64, units_2: 32, epochs: 100, batch: 32 },
      "ARIMA-LSTM": { max_p: 10, max_q: 10, time_step: 5, units: 48, dropout: 0.1, epochs: 50, batch: 32 }
    }
  };

  var ADVANCED_FIELDS = {
    "ARIMA": [
      { key: "max_p", label: "最大 p 阶", type: "number", min: 1, max: 30, step: 1 },
      { key: "max_q", label: "最大 q 阶", type: "number", min: 1, max: 30, step: 1 }
    ],
    "LSTM": [
      { key: "time_step", label: "滑窗步长", type: "number", min: 1, max: 30, step: 1 },
      { key: "units", label: "隐藏层单元数", type: "number", min: 8, max: 256, step: 8 },
      { key: "dropout", label: "Dropout", type: "number", min: 0, max: 0.5, step: 0.05 },
      { key: "epochs", label: "训练轮数", type: "number", min: 10, max: 200, step: 10 },
      { key: "batch", label: "批大小", type: "number", min: 8, max: 128, step: 8 }
    ],
    "SVR": [
      { key: "C", label: "惩罚系数 C", type: "number", min: 0.1, max: 1000, step: 0.1 },
      { key: "epsilon", label: "epsilon", type: "number", min: 0.01, max: 1, step: 0.01 },
      { key: "gamma", label: "gamma（scale/auto/数值）", type: "text" }
    ],
    "BP": [
      { key: "time_step", label: "滑窗步长", type: "number", min: 1, max: 60, step: 1 },
      { key: "units_1", label: "隐藏层1单元数", type: "number", min: 8, max: 256, step: 8 },
      { key: "units_2", label: "隐藏层2单元数", type: "number", min: 4, max: 128, step: 4 },
      { key: "epochs", label: "训练轮数", type: "number", min: 10, max: 200, step: 10 },
      { key: "batch", label: "批大小", type: "number", min: 8, max: 128, step: 8 }
    ],
    "ARIMA-LSTM": [
      { key: "max_p", label: "ARIMA 最大 p 阶", type: "number", min: 1, max: 30, step: 1 },
      { key: "max_q", label: "ARIMA 最大 q 阶", type: "number", min: 1, max: 30, step: 1 },
      { key: "time_step", label: "LSTM 滑窗步长", type: "number", min: 1, max: 30, step: 1 },
      { key: "units", label: "LSTM 单元数", type: "number", min: 8, max: 256, step: 8 },
      { key: "dropout", label: "LSTM Dropout", type: "number", min: 0, max: 0.5, step: 0.05 },
      { key: "epochs", label: "LSTM 训练轮数", type: "number", min: 10, max: 200, step: 10 },
      { key: "batch", label: "LSTM 批大小", type: "number", min: 8, max: 128, step: 8 }
    ]
  };

  // ---------- 工具 ----------
  function showError(msg) {
    $errorBanner.textContent = "错误：" + msg;
    $errorBanner.classList.remove("hidden");
  }
  function clearError() { $errorBanner.classList.add("hidden"); }
  function setStatus(s) { $statusLine.textContent = s || ""; }
  function setProgress(text, pct) {
    $progressText.textContent = text;
    $progressFill.style.width = pct + "%";
  }

  function showConfirm(message, onConfirm) {
    modalCallback = onConfirm || null;
    $modalMessage.textContent = message;
    $modalOverlay.classList.remove("hidden");
  }

  function hideModal() {
    $modalOverlay.classList.add("hidden");
    modalCallback = null;
  }

  // ---------- 滑杆 ----------
  $horizon.addEventListener("input", function () {
    var v = parseInt($horizon.value, 10);
    $horizonValue.textContent = v + " 天";
    if (v > 30) {
      $horizonWarning.classList.remove("hidden");
    } else {
      $horizonWarning.classList.add("hidden");
    }
  });

  // ---------- 快捷按钮 ----------
  document.querySelectorAll(".btn-quick").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $code.value = btn.getAttribute("data-code");
      $code.focus();
    });
  });

  // ---------- 输入校验 ----------
  function validateCode() {
    var v = $code.value.trim();
    if (!/^\d{6}$/.test(v)) {
      showError("请输入 6 位数字股票代码（如 003015）");
      return null;
    }
    return v;
  }

  // ---------- 真实接口（FastAPI） ----------
  function runPrediction(code, horizon, forceRetrain) {
    return new Promise(function (resolve, reject) {
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: code,
          horizon: horizon,
          force_retrain: !!forceRetrain,
          options: readAdvancedOptions()
        })
      }).then(function (r) {
        return r.json();
      }).then(function (data) {
        if (data.error) throw new Error(data.error);
        var jobId = data.job_id;
        pollJob(jobId, resolve, reject);
      }).catch(reject);
    });
  }

  function pollJob(jobId, resolve, reject) {
    var timer = setInterval(function () {
      fetch("/api/job/" + jobId).then(function (r) {
        return r.json();
      }).then(function (job) {
        if (job.error) { clearInterval(timer); reject(new Error(job.error)); return; }
        var pct = job.pct || 0;
        var detail = job.detail || "";
        if (job.status === "done" && job.result) {
          clearInterval(timer);
          setProgress("完成", 100);
          resolve(job.result);
        } else if (job.status === "error") {
          clearInterval(timer);
          reject(new Error(job.error || "训练失败"));
        } else {
          setProgress(detail + "（" + pct + "%）", pct);
        }
      }).catch(function (err) {
        clearInterval(timer);
        reject(err);
      });
    }, 1000);
  }

  // ---------- 历史记录 ----------
  function loadHistory() {
    try {
      var arr = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveHistory(list) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
    } catch (e) { /* localStorage 不可用时静默 */ }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c];
    });
  }

  function addHistory(code, name) {
    var list = loadHistory();
    var now = Date.now();
    var idx = -1;
    for (var i = 0; i < list.length; i++) {
      if (list[i].code === code) { idx = i; break; }
    }
    if (idx >= 0) {
      var item = list.splice(idx, 1)[0];
      item.name = name || item.name || "";
      item.t = now;
      list.unshift(item);
    } else {
      list.unshift({ code: code, name: name || "", t: now });
    }
    saveHistory(list.slice(0, 20));
    renderHistory();
  }

  function renderHistory() {
    var list = loadHistory();
    var body = "";
    if (!list.length) {
      body = '<div class="history-empty">暂无历史记录</div>';
    } else {
      var rows = "";
      list.forEach(function (item, i) {
        var name = item.name
          ? '<span class="history-name">' + escapeHtml(item.name) + "</span>"
          : "";
        rows += '<div class="history-item" data-index="' + i + '">' +
          '<span><span class="history-code">' + escapeHtml(item.code) + "</span>" + name + "</span>" +
          '<button class="history-del" data-index="' + i + '" title="删除该记录">×</button>' +
          "</div>";
      });
      body = '<div class="history-list">' + rows + "</div>" +
        '<div class="history-footer">' +
        '<button id="btn-clear-history" class="btn-clear-history">清空历史</button>' +
        "</div>";
    }
    $historyDropdown.innerHTML =
      '<div class="history-head"><b>历史记录</b>' +
      '<span class="history-count">共 ' + list.length + " 条</span></div>" + body;
  }

  function showHistory() {
    renderHistory();
    $historyDropdown.classList.remove("hidden");
  }

  function hideHistory() {
    $historyDropdown.classList.add("hidden");
  }

  // ---------- 高级选项 ----------
  function deepClone(o) {
    return JSON.parse(JSON.stringify(o));
  }

  function mergeAdvanced(obj) {
    var d = ADVANCED_DEFAULTS;
    var out = {
      max_rows: (typeof obj.max_rows === "number" && isFinite(obj.max_rows))
        ? obj.max_rows : d.max_rows,
      train_ratio: (typeof obj.train_ratio === "number" && isFinite(obj.train_ratio))
        ? obj.train_ratio : d.train_ratio,
      garch_seed: (typeof obj.garch_seed === "number" && isFinite(obj.garch_seed))
        ? obj.garch_seed : d.garch_seed,
      models: (Array.isArray(obj.models) && obj.models.length)
        ? obj.models.filter(function (m) { return d.models.indexOf(m) >= 0; })
        : d.models.slice(),
      params: {}
    };
    out.max_rows = Math.max(1000, Math.min(3000, Math.round(out.max_rows)));
    out.train_ratio = Math.max(0.5, Math.min(0.9, out.train_ratio));
    if (!out.models.length) out.models = d.models.slice();
    Object.keys(d.params).forEach(function (name) {
      var base = d.params[name];
      var given = (obj.params && obj.params[name]) || {};
      var p = {};
      Object.keys(base).forEach(function (k) {
        var v = (given[k] !== undefined && given[k] !== null) ? given[k] : base[k];
        var field = null;
        (ADVANCED_FIELDS[name] || []).forEach(function (f) {
          if (f.key === k) field = f;
        });
        if (field && field.type !== "text" && typeof v === "number" && isFinite(v)) {
          var lo = field.min !== undefined ? field.min : -Infinity;
          var hi = field.max !== undefined ? field.max : Infinity;
          v = Math.max(lo, Math.min(hi, v));
          if (Number.isInteger(base[k])) v = Math.round(v);
        }
        p[k] = v;
      });
      out.params[name] = p;
    });
    return out;
  }

  function advancedOptions() {
    try {
      var raw = localStorage.getItem(ADVANCED_KEY);
      if (raw) {
        var obj = JSON.parse(raw);
        if (obj && typeof obj === "object") return mergeAdvanced(obj);
      }
    } catch (e) { /* 忽略损坏数据 */ }
    return deepClone(ADVANCED_DEFAULTS);
  }

  function saveAdvancedOptions(opts) {
    try {
      localStorage.setItem(ADVANCED_KEY, JSON.stringify(opts));
    } catch (e) { /* localStorage 不可用时静默 */ }
  }

  function clampNum(v, lo, hi, fallback) {
    var n = parseFloat(v);
    if (isNaN(n)) return fallback;
    return Math.max(lo, Math.min(hi, n));
  }

  function readAdvancedOptions() {
    var opts = advancedOptions();
    var gen = $advancedGeneral;
    var maxRowsInput = gen.querySelector('[data-key="max_rows"] input');
    var ratioInput = gen.querySelector('[data-key="train_ratio"] input');
    var seedInput = gen.querySelector('[data-key="garch_seed"] input');
    if (maxRowsInput) {
      opts.max_rows = Math.round(clampNum(maxRowsInput.value, 1000, 3000, opts.max_rows));
    }
    if (ratioInput) {
      opts.train_ratio = clampNum(ratioInput.value, 50, 90, opts.train_ratio * 100) / 100;
    }
    if (seedInput) {
      var seed = parseInt(seedInput.value, 10);
      opts.garch_seed = isNaN(seed) ? opts.garch_seed : seed;
    }
    opts.models = Array.prototype.filter.call(
      gen.querySelectorAll('[data-key="models"] input[type="checkbox"]'),
      function (c) { return c.checked; }
    ).map(function (c) { return c.value; });
    if (!opts.models.length) opts.models = ADVANCED_DEFAULTS.models.slice();

    Object.keys(ADVANCED_FIELDS).forEach(function (name) {
      $advancedModels.querySelectorAll('[data-model="' + name + '"]').forEach(function (box) {
        var key = box.getAttribute("data-key");
        var input = box.querySelector("input");
        if (!input) return;
        var v = input.value;
        if (key === "gamma") {
          var num = parseFloat(v);
          opts.params[name][key] = isNaN(num) ? String(v) : num;
        } else {
          var n = parseFloat(v);
          opts.params[name][key] = isNaN(n)
            ? ADVANCED_DEFAULTS.params[name][key] : n;
        }
      });
    });
    return opts;
  }

  function fieldHtml(f, name, value) {
    var v = value === undefined ? "" : value;
    var attrs = ' type="' + (f.type || "number") + '" value="' + v + '"';
    if (f.min !== undefined) attrs += ' min="' + f.min + '"';
    if (f.max !== undefined) attrs += ' max="' + f.max + '"';
    if (f.step !== undefined) attrs += ' step="' + f.step + '"';
    return '<div class="advanced-field" data-model="' + name + '" data-key="' + f.key + '">' +
      "<label>" + f.label + "</label>" +
      "<input" + attrs + "></div>";
  }

  function buildAdvancedUI() {
    var opts = advancedOptions();
    var gen = "";
    gen += '<div class="advanced-group"><h3>通用参数</h3><div class="advanced-grid">';
    gen += '<div class="advanced-field" data-key="max_rows">' +
      "<label>数据天数上限</label>" +
      '<input type="number" min="1000" max="3000" step="100" value="' + opts.max_rows + '"></div>';
    gen += '<div class="advanced-field" data-key="train_ratio">' +
      "<label>训练比例（%）</label>" +
      '<input type="number" min="50" max="90" step="5" value="' + Math.round(opts.train_ratio * 100) + '"></div>';
    gen += '<div class="advanced-field" data-key="garch_seed">' +
      "<label>GARCH 随机种子</label>" +
      '<input type="number" step="1" value="' + opts.garch_seed + '"></div>';
    gen += "</div>";
    gen += '<div class="advanced-field" data-key="models">' +
      "<label>启用模型</label>" +
      '<div class="model-toggle-row">';
    ADVANCED_DEFAULTS.models.forEach(function (m) {
      var checked = opts.models.indexOf(m) >= 0 ? " checked" : "";
      var on = checked ? " on" : "";
      gen += '<label class="model-chip' + on + '"><input type="checkbox" value="' + m + '"' + checked +
        '><span>' + m + "</span></label>";
    });
    gen += "</div></div></div>";
    $advancedGeneral.innerHTML = gen;

    var modelsHtml = "";
    Object.keys(ADVANCED_FIELDS).forEach(function (name) {
      var grid = "";
      ADVANCED_FIELDS[name].forEach(function (f) {
        grid += fieldHtml(f, name, opts.params[name][f.key]);
      });
      modelsHtml += '<div class="advanced-group"><details><summary>' + name + "</summary>" +
        '<div class="advanced-grid">' + grid + "</div></details></div>";
    });
    $advancedModels.innerHTML = modelsHtml;
  }

  // ---------- 渲染 ----------
  function bestModel(metrics) {
    var names = Object.keys(metrics).filter(function (k) {
      return metrics[k] && isFinite(metrics[k].R2);
    });
    names.sort(function (a, b) {
      var ra = metrics[a].R2, rb = metrics[b].R2;
      if (rb !== ra) return rb - ra;
      return metrics[a].RMSE - metrics[b].RMSE;
    });
    return names[0];
  }

  function renderMetrics(metrics) {
    var best = bestModel(metrics);
    var order = ["ARIMA", "LSTM", "SVR", "BP", "ARIMA-LSTM"];
    var rows = "";
    order.forEach(function (name) {
      var m = metrics[name];
      if (!m) return;
      var isBest = name === best;
      var badge = isBest ? '<span class="badge-best">最优</span>' : "";
      rows += '<tr class="' + (isBest ? "best-row" : "") + '">' +
        "<td>" + name + "</td>" +
        "<td>" + m.MAE.toFixed(3) + "</td>" +
        "<td>" + m.MSE.toFixed(3) + "</td>" +
        "<td>" + m.RMSE.toFixed(3) + "</td>" +
        "<td>" + m.R2.toFixed(4) + "</td>" +
        "<td>" + badge + "</td></tr>";
    });
    $metricsBody.innerHTML = rows;
  }

  function colorsFor(name) {
    var map = {
      "实际值": "#24292f",
      "ARIMA": "#2ca02c",
      "LSTM": "#17becf",
      "SVR": "#1f77b4",
      "BP": "#9467bd",
      "ARIMA-LSTM": "#d62728"
    };
    return map[name] || "#888";
  }

  function renderTestChart(data) {
    var el = document.getElementById("chart-test");
    if (!chartTest) {
      chartTest = echarts.init(el);
    } else {
      chartTest.resize();
    }
    var series = [{
      name: "实际值", type: "line", data: data.test.actual,
      lineStyle: { width: 2.5, color: "#24292f" }, itemStyle: { color: "#24292f" }, symbol: "none", z: 5,
      emphasis: {
        lineStyle: { width: 3.5, shadowBlur: 10, shadowColor: "rgba(36, 41, 47, 0.45)" },
        itemStyle: { shadowBlur: 10, shadowColor: "rgba(36, 41, 47, 0.45)" }
      }
    }];
    var order = ["SVR", "ARIMA", "BP", "LSTM", "ARIMA-LSTM"];
    order.forEach(function (name) {
      var vals = data.test.models[name];
      if (!vals) return;
      var isHybrid = name === "ARIMA-LSTM";
      series.push({
        name: name, type: "line", data: vals,
        lineStyle: { width: isHybrid ? 2.6 : 1.3, opacity: isHybrid ? 1 : 0.85 },
        itemStyle: { color: colorsFor(name) },
        symbol: "none",
        z: isHybrid ? 4 : 2,
        emphasis: { focus: "series" }
      });
    });
    var bestName = bestModel(data.metrics);
    var bestIdx = -1;
    series.forEach(function (s, i) {
      if (s.name === bestName) bestIdx = i;
    });
    if (bestIdx > 0) {
      series[bestIdx].emphasis = {
        focus: "series",
        lineStyle: { width: 3.5, shadowBlur: 10, shadowColor: "rgba(31, 111, 235, 0.45)" },
        itemStyle: { shadowBlur: 10, shadowColor: "rgba(31, 111, 235, 0.45)" }
      };
    }
    highlightBest = bestIdx;
    if (!chartTest._emphasisBound) {
      chartTest._emphasisBound = true;
      chartTest.on("mouseover", function () {
        [0, highlightBest].forEach(function (idx) {
          if (idx >= 0) chartTest.dispatchAction({ type: "highlight", seriesIndex: idx });
        });
      });
      chartTest.on("globalout", function () {
        [0, highlightBest].forEach(function (idx) {
          if (idx >= 0) chartTest.dispatchAction({ type: "downplay", seriesIndex: idx });
        });
      });
    }
    chartTest.setOption({
      tooltip: { trigger: "axis" },
      legend: { top: 0, type: "scroll" },
      grid: { left: 60, right: 20, top: 40, bottom: 50 },
      xAxis: { type: "category", data: data.test.dates, axisLabel: { fontSize: 11 } },
      yAxis: { type: "value", name: "开盘价（元）", scale: true },
      dataZoom: [
        { type: "inside" },
        { type: "slider", height: 18, bottom: 8, start: 40, end: 100 }
      ],
      series: series
    }, true);
  }

  function renderForecastChart(data) {
    var el2 = document.getElementById("chart-forecast");
    if (!chartForecast) {
      chartForecast = echarts.init(el2);
    } else {
      chartForecast.resize();
    }
    var histLen = Math.min(60, data.history.open.length);
    var histDates = data.history.dates.slice(-histLen);
    var histVals = data.history.open.slice(-histLen);
    var fc = data.forecast;
    var forecastStartIndex = histDates.length;

    // 预测/置信带前面补空位，使序列从预测起点（虚线处）开始绘制
    var pad = [];
    for (var i = 0; i < forecastStartIndex; i++) pad.push(null);
    var fcValues = pad.concat(fc.values);
    var fcLower = pad.concat(fc.lower);
    var fcUpper = pad.concat(fc.upper);

    // 置信带宽度 = 上沿 - 下沿（stack 在下限之上堆叠，只染色 [下限, 上限] 区间）
    var bandWidth = fcLower.map(function (l, i) {
      var u = fcUpper[i];
      return (u === null || l === null) ? null : Math.max(0, u - l);
    });

    var series = [
      {
        name: "历史开盘价（近" + histLen + "日）", type: "line", data: histVals,
        lineStyle: { width: 2, color: "#57606a" }, itemStyle: { color: "#57606a" }, symbol: "none", z: 3
      }
    ];
    // 红色模拟路径（GARCH 蒙特卡洛）：只保留与灰色预测曲线最接近的一条
    var allPaths = fc.paths || [];
    var bestPath = null, bestDist = Infinity;
    allPaths.forEach(function (p) {
      var n = Math.min(p.length, fc.values.length);
      if (!n) return;
      var d = 0;
      for (var j = 0; j < n; j++) {
        var diff = p[j] - fc.values[j];
        d += diff * diff;
      }
      if (d < bestDist) {
        bestDist = d;
        bestPath = p;
      }
    });
    if (bestPath) {
      series.push({
        name: "模拟路径", type: "line", data: pad.concat(bestPath),
        lineStyle: { width: 1.5, color: "rgba(214, 39, 40, 0.85)" },
        itemStyle: { color: "rgba(214, 39, 40, 0.85)" },
        symbol: "none", z: 2, legendHoverLink: false
      });
    }
    series.push(
      {
        name: "置信区间", type: "line", data: fcLower, stack: "conf-band",
        lineStyle: { opacity: 0 }, symbol: "none", z: 1,
        areaStyle: { color: "rgba(87, 96, 106, 0)", opacity: 0 }
      },
      {
        name: "置信区间", type: "line", data: bandWidth, stack: "conf-band",
        lineStyle: { opacity: 0 }, symbol: "none", z: 1,
        areaStyle: { color: "rgba(87, 96, 106, 0.25)", opacity: 0.25 }
      },
      {
        name: "预测模型（" + fc.model + "）未来预测", type: "line", data: fcValues,
        lineStyle: { width: 3, color: "#9aa4af" }, itemStyle: { color: "#9aa4af" }, symbol: "circle", symbolSize: 4, z: 4,
        markLine: {
          silent: true, symbol: "none",
          label: { formatter: "预测起点", position: "insideEndTop", fontSize: 12, color: "#9a6700" },
          lineStyle: { type: "dashed", color: "#9a6700", width: 1.5 },
          data: [{ xAxis: forecastStartIndex }]
        }
      }
    );

    chartForecast.setOption({
      tooltip: { trigger: "axis" },
      legend: { top: 0, type: "scroll" },
      grid: { left: 60, right: 20, top: 40, bottom: 50 },
      xAxis: {
        type: "category", boundaryGap: false,
        data: histDates.concat(fc.dates),
        axisLabel: { fontSize: 11 }
      },
      yAxis: { type: "value", name: "开盘价（元）", scale: true },
      series: series
    }, true);

    var volNote = fc.path_method === "garch" ? "GARCH(1,1) 时变波动率" :
      (fc.path_method === "fallback" ? "历史波动率（arch 未安装）" : "模型置信区间");
    $forecastNote.textContent = "预测模型：" + fc.model + " · 预测期 " + fc.dates[0] + " ~ " + fc.dates[fc.dates.length - 1] +
      " · 波动来源：" + volNote +
      " · 预测区间最高 " + Math.max.apply(null, fc.upper).toFixed(2) + " 元 / 最低 " + Math.min.apply(null, fc.lower).toFixed(2) + " 元";
  }

  function renderDataInfo(data) {
    var s = data.stock;
    var warn = s.warn ? '<li style="color:#d1242f">警告：' + s.warn + "</li>" : "";
    var adv = readAdvancedOptions();
    var trainPct = Math.round(adv.train_ratio * 100);
    if ($footerSource) {
      $footerSource.textContent = s.cache;
    }
    $dataInfo.innerHTML =
      "<li>股票：" + s.name + "（" + s.code + "）</li>" +
      "<li>数据区间：" + s.start + " ~ " + s.end + "（共 " + s.samples + " 个交易日，最新数据截至 " + s.end + "）</li>" +
      "<li>训练数据上限：最近 " + adv.max_rows + " 个交易日</li>" +
      "<li>训练/测试划分：" + trainPct + "% / " + (100 - trainPct) + "%（时间顺序，不打乱）</li>" +
      "<li>数据来源：" + s.cache + "</li>" +
      warn +
      (data.note ? "<li>说明：" + data.note + "</li>" : "");
  }

  function renderAll(data) {
    renderMetrics(data.metrics);
    renderDataInfo(data);
    $resultArea.classList.remove("hidden");
    // 等待容器可见后再初始化图表，否则 ECharts 宽度为 0 导致线条挤在一起
    requestAnimationFrame(function () {
      renderTestChart(data);
      renderForecastChart(data);
    });
  }

  // ---------- 主流程 ----------
  function start(code, horizon, forceRetrain) {
    if (state.busy) return;
    clearError();
    var code = validateCode();
    if (!code) return;

    state.busy = true;
    $btn.disabled = true;
    $retrain.disabled = true;
    $retrain.classList.add("hidden");
    $progressArea.classList.remove("hidden");
    $progressSpinner.classList.remove("hidden");
    setProgress("开始…", 2);
    setStatus((forceRetrain ? "正在重新训练 " : "正在为 ") + code + " 训练模型，请稍候（预计 1~3 分钟）…");
    $resultArea.classList.add("hidden");

    runPrediction(code, horizon, forceRetrain).then(function (data) {
      setStatus("完成。预测天数 " + horizon + " 天。");
      $progressArea.classList.add("hidden");
      $progressSpinner.classList.add("hidden");
      state.busy = false;
      $btn.disabled = false;
      $retrain.disabled = false;
      renderAll(data);
      if (data.stock && data.stock.code) {
        addHistory(data.stock.code, data.stock.name || "");
      }
      // 命中缓存时提供强制重训入口
      if (data.stock && data.stock.cache === "本地缓存") {
        $retrain.classList.remove("hidden");
      }
    }).catch(function (err) {
      state.busy = false;
      $btn.disabled = false;
      $retrain.disabled = false;
      $retrain.classList.add("hidden");
      $progressArea.classList.add("hidden");
      $progressSpinner.classList.add("hidden");
      setStatus("");
      showError(err && err.message ? err.message : "预测失败，请稍后重试");
    });
  }

  $btn.addEventListener("click", function () {
    start($code.value.trim(), parseInt($horizon.value, 10), false);
  });

  $retrain.addEventListener("click", function () {
    start($code.value.trim(), parseInt($horizon.value, 10), true);
  });

  // 回车触发
  $code.addEventListener("keydown", function (e) {
    if (e.key === "Enter") $btn.click();
  });

  // ---------- 历史记录交互 ----------
  $code.addEventListener("focus", showHistory);
  $code.addEventListener("keydown", function (e) {
    if (e.key === "Escape") hideHistory();
  });
  $historyDropdown.addEventListener("click", function (e) {
    var del = e.target.closest(".history-del");
    if (del) {
      e.stopPropagation();
      var idx = parseInt(del.getAttribute("data-index"), 10);
      var list = loadHistory();
      if (idx >= 0 && idx < list.length) {
        list.splice(idx, 1);
        saveHistory(list);
        renderHistory();
      }
      return;
    }
    var clearBtn = e.target.closest("#btn-clear-history");
    if (clearBtn) {
      showConfirm("确定清空所有历史记录吗？", function () {
        saveHistory([]);
        renderHistory();
      });
      return;
    }
    var item = e.target.closest(".history-item");
    if (item) {
      var idx2 = parseInt(item.getAttribute("data-index"), 10);
      var list2 = loadHistory();
      if (list2[idx2]) {
        $code.value = list2[idx2].code;
        hideHistory();
      }
    }
  });
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".code-field")) hideHistory();
  });

  $modalConfirm.addEventListener("click", function () {
    var cb = modalCallback;
    hideModal();
    if (cb) cb();
  });
  $modalCancel.addEventListener("click", hideModal);
  $modalOverlay.addEventListener("click", function (e) {
    if (e.target === $modalOverlay) hideModal();
  });

  // ---------- 高级选项交互 ----------
  $btnAdvanced.addEventListener("click", function () {
    $advancedPanel.classList.toggle("hidden");
  });
  $btnResetAdvanced.addEventListener("click", function () {
    if (!confirm("确定恢复所有高级参数为默认值吗？")) return;
    saveAdvancedOptions(deepClone(ADVANCED_DEFAULTS));
    buildAdvancedUI();
  });
  $advancedGeneral.addEventListener("input", function () {
    saveAdvancedOptions(readAdvancedOptions());
  });
  $advancedGeneral.addEventListener("change", function (e) {
    var chip = e.target && e.target.closest ? e.target.closest(".model-chip") : null;
    if (chip && e.target.type === "checkbox") {
      chip.classList.toggle("on", e.target.checked);
    }
  });
  $advancedModels.addEventListener("input", function () {
    saveAdvancedOptions(readAdvancedOptions());
  });

  // ---------- 初始化 ----------
  buildAdvancedUI();
  renderHistory();

  // 窗口大小变化时重绘
  window.addEventListener("resize", function () {
    if (chartTest) chartTest.resize();
    if (chartForecast) chartForecast.resize();
  });
})();
