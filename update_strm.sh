#!/bin/bash

# ================= 配置区 =================
STRM_FILE="strm.txt"
ZIP_FILE="strm.zip"
LOCAL_ZIP="local_strm_list.zip"  # 现在指向 zip 文件
SIGN="?sign=SIGN_STR"
SLEEP_TIME=21600 # 6小时
# ==========================================

# 1. 杀掉旧的实例，排除当前进程 PID ($$)
echo "正在检查并清理旧进程..."
pgrep -f "$(basename "$0")" | grep -v $$ | xargs kill 2>/dev/null 2>&1

# 确保在脚本所在目录运行
# cd "$(dirname "$0")"

# 检查依赖工具
for cmd in python3 unzip zip git; do
    if ! command -v $cmd &> /dev/null; then
        echo "错误: 未找到命令 $cmd，请先安装。"
        exit 1
    fi
done

while true; do
    echo "[$(date)] 开始更新任务..."

    # 2. 运行 Python 爬虫
    if python3 "$(dirname "$0")/strm_crawler.py"; then
        
        # 3. 处理本地 ZIP 文件并追加进 strm.txt
        if [ -f "$LOCAL_ZIP" ]; then
            # unzip -p 代表将解压后的内容直接输出到 stdout
            # 这样不需要产生临时文件，直接追加到 strm.txt
            unzip -p "$LOCAL_ZIP" >> "$STRM_FILE"
            echo "已从 $LOCAL_ZIP 提取并合并数据。"
        fi

        # 4. 处理签名和去重
        if [ -f "$STRM_FILE" ]; then
            # 给没有签名的行加签名 (幂等处理)
            sed -i "/?sign=/! s|$|$SIGN|" "$STRM_FILE"
            # 排序并去重
            sort -u "$STRM_FILE" -o "$STRM_FILE"
            
            echo "所有条目已完成处理。当前总行数: $(wc -l < "$STRM_FILE")"

            # 5. 压缩成 ZIP 供发布
            zip -qj "$ZIP_FILE" "$STRM_FILE"
            echo "已生成压缩包: $ZIP_FILE"

            # 6. Git 提交与推送
            git add "$ZIP_FILE" "$STRM_FILE" # 建议两个都 add，确保同步
            
            if ! git diff --cached --quiet; then
                git commit -m "自动更新strm，时间：$(date +'%Y-%m-%d %H:%M:%S')"
                if git push; then
                    echo "Git 推送成功。"
                else
                    echo "警告：Git 推送失败，可能是网络问题，将在下次循环重试。"
                fi
            else
                echo "文件内容无变化，跳过提交。"
            fi
        fi

    else
        echo "警告：爬虫运行失败，跳过本次更新。"
    fi

    echo "[$(date)] 进入休眠，等待下一次循环..."
    sleep "$SLEEP_TIME"
done