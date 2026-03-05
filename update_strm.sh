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
    echo -e "\n[!] 收到中断信号，正在深度清理子进程..."
    
    # 获取当前脚本的进程组 ID (PGID) 并杀掉整个组
    # 这样可以确保 Python 爬虫、unzip、git 等子进程无一漏网
    # 使用 -$$ 表示当前进程组
    trap - SIGINT SIGTERM # 清除 trap 防止死循环
    rm -f "$PID_FILE"
    kill -9 -$$ 2>/dev/null
}

# 注册信号捕捉
trap cleanup SIGINT SIGTERM

# 1. 杀掉旧的实例 (包括它们的子进程)
echo "正在检查旧进程..."
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    # 获取旧进程的进程组 ID 并杀掉全组
    OLD_PGID=$(ps -o pgid= -p "$OLD_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$OLD_PGID" ]; then
        echo "发现旧进程组 ($OLD_PGID)，正在深度清理..."
        kill -9 -"$OLD_PGID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
fi

# 记录当前进程 PID
echo $$ > "$PID_FILE"
echo "[+] 当前进程 PID ($$) 已记录"

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

    git pull >/dev/null 2>&1
    rm -f "$STRM_FILE" >/dev/null 2>&1

    # 运行 Python 爬虫 (后台运行并等待，方便捕捉中断)
    python3 "$(dirname "$0")/strm_crawler.py" &
    PYTHON_PID=$!
    wait $PYTHON_PID
    
    # 检查 python 退出状态
    if [ $? -eq 0 ]; then
        # 处理本地 ZIP 文件
        if [ -f "$LOCAL_ZIP" ]; then
            echo "[INFO] 正在处理本地 ZIP 数据..."
            # 1. 提取本地 ZIP 内容到临时文件
            unzip -p "$LOCAL_ZIP" > .tmp_local_content
            
            # 2. 过滤：保留那些“不在 STRM_FILE 中”的行
            # -F: 固定字符串, -v: 反向匹配, -x: 整行匹配, -f: 从文件读取匹配模板
            grep -Fvxf "$STRM_FILE" .tmp_local_content > .tmp_filtered_content || true
            
            # 3. 比较行数，判断是否有重复项被删除
            # 因为local_strm_list.zip可能是全量文件生成的，需要把爬虫服务器上有的删除
            ORIG_COUNT=$(wc -l < .tmp_local_content)
            NEW_COUNT=$(wc -l < .tmp_filtered_content)
            
            if [ "$NEW_COUNT" -lt "$ORIG_COUNT" ]; then
                echo "[UPDATE] LOCAL_ZIP 中发现 $((ORIG_COUNT - NEW_COUNT)) 条重复项，正在清理..."
                # 重新打包 LOCAL_ZIP (内部文件名固定为 local_strm_list.txt)
                mv .tmp_filtered_content local_strm_list.txt
                zip -qj "$LOCAL_ZIP" local_strm_list.txt
                rm local_strm_list.txt
                LOCAL_CHANGED=true
            else
                rm .tmp_filtered_content
            fi

            # 4. 将本地内容（去重前或去重后均可，因为后面有 sort -u）合并到主文件
            cat .tmp_local_content >> "$STRM_FILE"
            rm .tmp_local_content
            echo "[OK] 已合并本地数据。"
        fi

        # 处理签名和去重
        if [ -f "$STRM_FILE" ]; then
            sed -i "/?sign=/! s|$|$SIGN|" "$STRM_FILE"
            sort -u "$STRM_FILE" -o "$STRM_FILE"
            
            echo "[INFO] 当前总行数: $(wc -l < "$STRM_FILE")"

            zip -qj "$ZIP_FILE" "$STRM_FILE"
            
            # --- 修改部分：Git 提交逻辑 ---
            git add "$ZIP_FILE"
            # 如果 LOCAL_ZIP 有变化，也加入暂存区
            if [ "$LOCAL_CHANGED" = true ]; then
                git add "$LOCAL_ZIP"
                echo "[INFO] LOCAL_ZIP 已加入 Git 提交队列。"
            fi

            if ! git diff --cached --quiet; then
                git commit -m "自动更新strm，时间：$(date +'%Y-%m-%d %H:%M:%S')"
                git push && echo "[SUCCESS] Git 推送成功。" || echo "[ERROR] 推送失败。"
            else
                echo "[SKIP] 无内容变化。"
            fi
        fi
    else
        # 如果 python 是被 kill 掉的，$? 通常不为 0
        echo "[INFO] 爬虫进程已结束或被中断。"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进入休眠..."
    sleep "$SLEEP_TIME" & wait $!
done
