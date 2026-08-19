#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  股票预测系统 - ARIMA/LSTM/SVR/BP/混合"
echo "=========================================="
echo ""

# 检查依赖
python -c "import fastapi, uvicorn, tensorflow, pmdarima" 2>/dev/null || {
    echo "[提示] 未检测到全部依赖，正在安装（请确保已激活 Python 3.8~3.10 环境）..."
    python -m pip install -r requirements.txt
}

echo "[启动] 服务地址：http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
