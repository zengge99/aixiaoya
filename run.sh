#!/bin/bash

# --- 配置区 ---
webdav_server=http://113.5.22.166:5678/dav
webdav_user="guest"
webdav_password="guest_Api789"
webdav_port=65432

index_server=http://113.5.22.166:5678
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

# --- 脚本路径解析 ---
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

# --- 进程管理逻辑 ---
PIDS=() 

# 停止先前运行的残留进程
stop_existing() {
    echo "[!] 检查并清理旧的后台进程..."
    
    # 1. 根据二进制名称清理 (匹配所有架构版本)
    # pkill -f 匹配完整命令行，比 killall 更强大
    pkill -f "movie_extractor_linux_" >/dev/null 2>&1
    pkill -f "webdav_linux_" >/dev/null 2>&1
    
    # 2. 根据端口清理 (防止进程名变了但端口仍被占用)
    # 尝试使用 fuser 杀掉占用端口的进程（如果系统安装了 psmisc）
    fuser -k 8889/tcp >/dev/null 2>&1
    fuser -k ${webdav_port}/tcp >/dev/null 2>&1
    
    sleep 1 # 等待进程完全退出
}

# 当前运行中的清理函数（Ctrl+C 时触发）
cleanup() {
    echo -e "\n\033[31m[!] 接收到中断信号，正在清理当前后台进程...\033[0m"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1
        fi
    done
    echo "[+] 清理完成，退出。"
    exit 0
}

# 捕获信号
trap cleanup SIGINT SIGTERM

# --- 执行开始 ---

# 1. 彻底清理旧进程
stop_existing

# 2. 准备权限
chmod 755 "movie_extractor_linux_$arch" "webdav_linux_$arch" "getalist_linux_$arch"

# 3. 启动后台服务
echo "[+] 正在启动后台服务..."

# 启动 movie_extractor (端口 8889)
./movie_extractor_linux_$arch --srv 8889 >/dev/null 2>&1 &
PIDS+=($!)

# 启动 webdav (端口 $webdav_port)
touch fake.txt
./webdav_linux_$arch --file "*.txt" --url "$webdav_server" --user "$webdav_user" --password "$webdav_password" --port "$webdav_port" --obfuscate &
PIDS+=($!)

# 验证后台进程是否成功启动
sleep 2
for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        echo -e "\033[31m[错误] 后台进程 $pid 启动失败，请检查配置或权限。\033[0m"
        exit 1
    fi
done

echo "[+] 后台服务已就绪 (PIDs: ${PIDS[*]})"

# 4. 执行循环任务
for task in "${tasks[@]}"; do
    url="$index_server${task%%|*}"
    output="${task##*|}"
    
    echo "----------------------------------------"
    echo "正在处理: $url"
    echo "保存到: $output"
    
    ./getalist_linux_$arch --url "$url" --user "$index_user" --password "$index_password" --output "$output"
    
    if [ $? -ne 0 ]; then
        echo "[!] getalist 任务中断"
        cleanup
    fi
done

echo "----------------------------------------"
echo "[+] 所有任务处理完成！"

# 正常退出提示
read -t 5 -p "脚本任务已完成，后台服务将继续运行。按 Ctrl+C 停止服务，或等待 5 秒自动退出脚本..." || echo ""