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
  var $statusLine = document.getElementById("status-line");
  var $resultArea = document.getElementById("result-area");
  var $errorBanner = document.getElementById("error-banner");
  var $metricsBody = document.querySelector("#metrics-table tbody");
  var $dataInfo = document.getElementById("data-info");
  var $forecastNote = document.getElementById("forecast-note");

  var chartTest = null, chartForecast = null;

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
        body: JSON.stringify({ code: code, horizon: horizon, force_retrain: !!forceRetrain })
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
      lineStyle: { width: 2.5, color: "#24292f" }, itemStyle: { color: "#24292f" }, symbol: "none", z: 5
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
      },
      {
        name: "置信区间", type: "line", data: fcLower, stack: "conf-band",
        lineStyle: { opacity: 0 }, symbol: "none", z: 1,
        areaStyle: { color: "rgba(214, 39, 40, 0)", opacity: 0 }
      },
      {
        name: "置信区间", type: "line", data: bandWidth, stack: "conf-band",
        lineStyle: { opacity: 0 }, symbol: "none", z: 1,
        areaStyle: { color: "rgba(214, 39, 40, 0.28)", opacity: 0.28 }
      },
      {
        name: "最优模型（" + fc.model + "）未来预测", type: "line", data: fcValues,
        lineStyle: { width: 3, color: "#d62728" }, itemStyle: { color: "#d62728" }, symbol: "circle", symbolSize: 4, z: 4,
        markLine: {
          silent: true, symbol: "none",
          label: { formatter: "预测起点", position: "insideEndTop", fontSize: 12, color: "#9a6700" },
          lineStyle: { type: "dashed", color: "#9a6700", width: 1.5 },
          data: [{ xAxis: forecastStartIndex }]
        }
      }
    ];

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

    $forecastNote.textContent = "最优模型：" + fc.model + "（R² 最高）· 预测期 " + fc.dates[0] + " ~ " + fc.dates[fc.dates.length - 1] +
      " · 预测区间最高 " + Math.max.apply(null, fc.upper).toFixed(2) + " 元 / 最低 " + Math.min.apply(null, fc.lower).toFixed(2) + " 元";
  }

function renderDataInfo(data) {
    var s = data.stock;
    var warn = s.warn ? '<li style="color:#d1242f">警告：' + s.warn + "</li>" : "";
    $dataInfo.innerHTML =
      "<li>股票：" + s.name + "（" + s.code + "）</li>" +
      "<li>数据区间：" + s.start + " ~ " + s.end + "（共 " + s.samples + " 个交易日，最新数据截至 " + s.end + "）</li>" +
      "<li>训练/测试划分：80% / 20%（时间顺序，不打乱）</li>" +
      "<li>状态：" + s.cache + "</li>" +
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
    setProgress("开始…", 2);
    setStatus((forceRetrain ? "正在重新训练 " : "正在为 ") + code + " 训练模型，请稍候（预计 1~3 分钟）…");
    $resultArea.classList.add("hidden");

    runPrediction(code, horizon, forceRetrain).then(function (data) {
      setStatus("完成。预测天数 " + horizon + " 天。");
      $progressArea.classList.add("hidden");
      state.busy = false;
      $btn.disabled = false;
      $retrain.disabled = false;
      renderAll(data);
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

  // 窗口大小变化时重绘
  window.addEventListener("resize", function () {
    if (chartTest) chartTest.resize();
    if (chartForecast) chartForecast.resize();
  });
})();
