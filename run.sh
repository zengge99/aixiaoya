#!/bin/bash

# --- 配置区 ---
webdav_server=http://113.65.22.166:5678
webdav_user="guest"
webdav_password="guest_Api789"
webdav_port=65432

index_server=http://113.65.22.166:5678
index_user="guest"
index_password="guest_Api789"

tasks=(
    "/🏷️我的115分享|115share.txt"
    "/每日更新|daily.txt"
    "/电影|dy.txt"
    "/电视剧|dsj.txt"
    "/综艺|zy.txt"
    "/纪录片|jlp.txt"
    "/整理中|zlz.txt"
)

# --- 脚本路径解析 (保留原逻辑) ---
prog="$0"
while [ -h "${prog}" ]; do
    newProg=`/bin/ls -ld "${prog}"`
    newProg=`expr "${newProg}" : ".* -> \(.*\)$"`
    if expr "x${newProg}" : 'x/' >/dev/null; then
        prog="${newProg}"
    else
        progdir=`dirname "${prog}"`
        prog="${progdir}/${newProg}"
    fi
done
cd "$(dirname "${prog}")"

# --- 架构检测 ---
machine=$(uname -m)
if [[ "$machine" == *"arm"* || "$machine" == *"aarch"* ]]; then
    arch="arm64"
else
    arch="amd64"
fi

# --- 变量与清理函数 ---
PIDS=() # 用于存放后台进程ID

cleanup() {
    echo -e "\n\033[31m[!] 接收到中断信号，正在清理后台进程...\033[0m"
    # 杀掉保存在数组里的后台 PID
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1
        fi
    done
    echo "[+] 清理完成，退出。"
    exit 0
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap cleanup SIGINT SIGTERM

# --- 启动前准备 ---
chmod 755 "movie_extractor_linux_$arch" "webdav_linux_$arch" "getalist_linux_$arch"

# 强力清理旧进程（可选）
killall "movie_extractor_linux_$arch" "webdav_linux_$arch" >/dev/null 2>&1

# --- 启动后台服务 ---

# 1. 启动 movie_extractor
./movie_extractor_linux_$arch --srv 8889 >/dev/null 2>&1 &
PIDS+=($!) # 记录 PID

# 2. 启动 webdav
touch fake.txt
./webdav_linux_$arch --file "*.txt" --url "$webdav_server" --user "$webdav_user" --password "$webdav_password" --port "$webdav_port" --obfuscate &
PIDS+=($!) # 记录 PID

echo "[+] 后台服务已启动 (PIDs: ${PIDS[*]})"

# --- 执行循环任务 ---
for task in "${tasks[@]}"; do
    url="$index_server${task%%|*}"
    output="${task##*|}"
    
    echo "----------------------------------------"
    echo "正在处理: $url"
    echo "保存到: $output"
    
    # 执行主任务
    ./getalist_linux_$arch --url "$url" --user "$index_user" --password "$index_password" --output "$output"
    
    # 检查上个命令状态，如果 getalist 被 Ctrl+C 也会中断循环
    if [ $? -ne 0 ]; then
        cleanup
    fi
done

echo "----------------------------------------"
echo "[+] 所有任务处理完成！"

# 脚本正常结束前，询问是否保留后台进程
read -t 5 -p "脚本即将退出，后台服务将继续运行。若要立即停止请按 Ctrl+C (5秒后自动后台化)..." || echo ""