#!/bin/bash

# 1. 配置变量
FILE1="movie_ner_bert.onnx"
FILE2="movie_ner_bert.onnx.data"
TEMP_ZIP="movie_ner_bert_model.zip"
CHUNK_SIZE="49M"
PREFIX="movie_ner_bert_model.zip.part"

# 2. 检查文件是否存在
if [ ! -f "$FILE1" ] || [ ! -f "$FILE2" ]; then
    echo "错误: 找不到 $FILE1 或 $FILE2"
    exit 1
fi

# 检查是否安装了 zip
if ! command -v zip &> /dev/null; then
    echo "错误: 未安装 zip，请先安装 (例如: sudo apt install zip)"
    exit 1
fi

echo "正在将文件压缩为 $TEMP_ZIP..."

# 3. 创建 ZIP 压缩包
# -q: 静默模式
# -j: 仅存储文件，不存储目录路径（可选）
zip -q "$TEMP_ZIP" "$FILE1" "$FILE2"

echo "正在拆分压缩包..."

# 4. 拆分压缩包
split -b $CHUNK_SIZE -d "$TEMP_ZIP" "$PREFIX"

# 5. 立即删除临时的庞大 ZIP 文件（避免占用本地空间和误传）
rm "$TEMP_ZIP"

echo "拆分完成，生成的文件如下："
ls -lh ${PREFIX}*

# 6. 更新 .gitignore
for f in "$FILE1" "$FILE2" "$TEMP_ZIP"; do
    if ! grep -q "$f" .gitignore 2>/dev/null; then
        echo "$f" >> .gitignore
    fi
done

# 7. Git 操作
echo "正在推送到 GitHub..."
git add "${PREFIX}*" .gitignore
git commit -m "Add split zip parts of movie_ner_bert model"
git push

echo "全部完成！"