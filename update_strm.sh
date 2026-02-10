#!/bin/bash

# ================= 配置区 =================
STRM_FILE="strm.txt"
ZIP_FILE="strm.zip"
LOCAL_ZIP="local_strm_list.zip"
SIGN="?sign=SIGN_STR"
SLEEP_TIME=21600 # 6小时
PID_FILE="/tmp/strm_updater.pid"
# ==========================================

# --- 核心函数：清理并退出 ---
cleanup() {
    echo -e "\n[!] 收到中断信号或脚本结束，正在清理..."
    rm -f "$PID_FILE"
    echo "[+] PID 文件已移除，安全退出。"
    exit 0
}

# 注册信号捕捉：Ctrl+C (SIGINT), Kill (SIGTERM), 脚本正常退出 (EXIT)
trap cleanup SIGINT SIGTERM EXIT

# 1. 杀掉旧的实例
echo "正在检查旧进程..."
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    # 检查该 PID 是否真的在运行
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "发现正在运行的旧实例 (PID: $OLD_PID)，正在强制终止..."
        kill -9 "$OLD_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
fi

# 记录当前进程 PID
echo $$ > "$PID_FILE"
echo "[+] 当前进程 PID ($$) 已记录至 $PID_FILE"

# 2. 检查依赖工具
for cmd in python3 unzip zip git; do
    if ! command -v $cmd &> /dev/null; then
        echo "错误: 未找到命令 $cmd，请先安装。"
        exit 1
    fi
done

# 3. 主循环
while true; do
    echo "-------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始更新任务..."

    # 运行 Python 爬虫
    if python3 "$(dirname "$0")/strm_crawler.py"; then
        
        # 处理本地 ZIP 文件
        if [ -f "$LOCAL_ZIP" ]; then
            unzip -p "$LOCAL_ZIP" >> "$STRM_FILE"
            echo "[OK] 已从 $LOCAL_ZIP 提取数据。"
        fi

        # 处理签名和去重
        if [ -f "$STRM_FILE" ]; then
            # 幂等处理签名
            sed -i "/?sign=/! s|$|$SIGN|" "$STRM_FILE"
            # 排序去重
            sort -u "$STRM_FILE" -o "$STRM_FILE"
            
            echo "[INFO] 处理完成。当前总行数: $(wc -l < "$STRM_FILE")"

            # 压缩
            zip -qj "$ZIP_FILE" "$STRM_FILE"
            echo "[OK] 已生成压缩包: $ZIP_FILE"

            # Git 提交
            git add "$ZIP_FILE"
            if ! git diff --cached --quiet; then
                git commit -m "自动更新strm，时间：$(date +'%Y-%m-%d %H:%M:%S')"
                if git push; then
                    echo "[SUCCESS] Git 推送成功。"
                else
                    echo "[ERROR] Git 推送失败，稍后重试。"
                fi
            else
                echo "[SKIP] 内容无变化，跳过提交。"
            fi
        fi
    else
        echo "[ERROR] 爬虫运行失败，跳过本次更新。"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 任务结束，进入休眠..."
    echo "提示：按下 Ctrl+C 可停止脚本。"
    
    # 这里的 wait $! 配合后台 sleep 可以让脚本立即响应 Ctrl+C
    sleep "$SLEEP_TIME" & wait $!
done
