#!/bin/bash

# --- 配置参数 ---
BASE_URL="http://emby.xiaoya.pro"
OUTPUT_FILE="strm.txt"
THREADS=10
TMP_DIR="./crawl_tmp"

# 创建临时工作目录
mkdir -p "$TMP_DIR"
# 清空输出文件
> "$OUTPUT_FILE"

# --- 核心函数：URL 解码 (纯 Bash) ---
urldecode() {
    local data="${1//+/ }"
    printf '%b' "${data//%/\\x}"
}

export -f urldecode

# --- 核心函数：处理单个文件或目录 ---
# 参数 $1: 相对路径 (例如 /电影/)
process_item() {
    local rel_path="$1"
    local full_url="${BASE_URL}${rel_path}"
    
    # 如果是 .strm 文件
    if [[ "$rel_path" == *.strm ]]; then
        # 1. 获取内容 (去除换行符)
        local content
        content=$(curl -s -L "$full_url" | tr -d '\r\n')
        
        # 2. 解码路径并移除前缀
        local decoded_path
        decoded_path=$(urldecode "$rel_path")
        
        # 3. 写入文件 (使用独占锁确保多线程安全)
        # 格式: 路径#内容
        printf "%s#%s\n" "$decoded_path" "$content" >> "$OUTPUT_FILE"
        echo "已抓取: $decoded_path" >&2
        
    else
        # 如果是目录，解析 HTML 提取更多链接
        local html
        html=$(curl -s -L "$full_url")
        
        # 提取 href，排除上级目录、带参数链接、绝对路径
        echo "$html" | grep -oP 'href="\K[^"?/][^"]*' | while read -r link; do
            local next_rel
            # 补全路径
            if [[ "$rel_path" == */ ]]; then
                next_rel="${rel_path}${link}"
            else
                next_rel="${rel_path}/${link}"
            fi
            
            # 判断是否是目录 (在 HTML 中 href 后面通常带 /)
            if echo "$html" | grep -qP "href=\"$link/\""; then
                echo "${next_rel}/" >> "$TMP_DIR/next_generation.txt"
            else
                echo "${next_rel}" >> "$TMP_DIR/next_generation.txt"
            fi
        done
    fi
}

export -f process_item
export BASE_URL OUTPUT_FILE TMP_DIR

# --- 主逻辑：BFS 逐层扫描 ---

# 初始任务：根目录
echo "/" > "$TMP_DIR/todo.txt"
touch "$TMP_DIR/visited.txt"

echo "开始爬取 $BASE_URL ..."
start_time=$(date +%s)

while [ -s "$TMP_DIR/todo.txt" ]; do
    # 记录已访问，防止死循环
    cat "$TMP_DIR/todo.txt" >> "$TMP_DIR/visited.txt"
    
    # 清空下一层任务
    > "$TMP_DIR/next_generation.txt"
    
    # 并发处理当前层的任务
    # -I{} 替换符, -P 10 并发10线程
    cat "$TMP_DIR/todo.txt" | xargs -I {} -P "$THREADS" bash -c 'process_item "$@"' _ "{}"
    
    # 生成下一层任务列表：去重并排除已访问过的目录
    if [ -f "$TMP_DIR/next_generation.txt" ]; then
        sort -u "$TMP_DIR/next_generation.txt" | grep -Fxf "$TMP_DIR/visited.txt" -v > "$TMP_DIR/todo.txt"
    else
        > "$TMP_DIR/todo.txt"
    fi
    
    count=$(wc -l < "$TMP_DIR/todo.txt")
    echo ">> 当前层级扫描完成，下一波任务数: $count"
done

# 清理并统计
rm -rf "$TMP_DIR"
end_time=$(date +%s)
echo "--------------------------------------"
echo "爬取完成！结果已存入 $OUTPUT_FILE"
echo "总耗时: $(($end_time - $start_time)) 秒"