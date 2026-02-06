#!/bin/bash

# ================= 配置区 =================
STRM_FILE="strm.txt"
LOCAL_LIST="local_strm_list.txt"
SIGN="?sign=SIGN_STR"
SLEEP_TIME=21600
# ==========================================

# 1. 杀掉旧的实例，排除当前进程 PID ($$)
# 使用 pgrep -f 匹配脚本名，排除自己后 kill
echo "正在检查并清理旧进程..."
pgrep -f "$(basename "$0")" | grep -v $$ | xargs kill 2>/dev/null

while true; do
    echo "[$(date)] 开始更新任务..."

    # 2. 运行 Python 爬虫
    if python strm_crawler.py; then
        
        # 3. 将 local_strm_list.txt 的内容追加进 strm.txt
        if [ -f "$LOCAL_LIST" ]; then
            cat "$LOCAL_LIST" >> "$STRM_FILE"
            echo "已合并本地列表。"
        fi

        # 4. 处理签名
        if [ -f "$STRM_FILE" ]; then
            sed -i "/?sign=/! s|$|$SIGN|" "$STRM_FILE"
            
            # 去重
            sort -u "$STRM_FILE" -o "$STRM_FILE"
            
            echo "所有条目已完成签名处理。总行数: $(wc -l < "$STRM_FILE")"
        fi

    else
        echo "警告：爬虫运行失败，跳过本次更新。"
    fi

    echo "进入休眠，${SLEEP_TIME}秒后再次执行..."
    sleep "$SLEEP_TIME"
done