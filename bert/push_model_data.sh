#!/bin/bash

# 1. 配置变量
FILE1="movie_ner_bert.onnx"
FILE2="movie_ner_bert.onnx.data"
CHUNK_SIZE="49M"  # 保持在 50M 以下，避免 GitHub 警告
PREFIX="movie_ner_bert_model.tar.gz.part" # 统一的压缩包前缀

# 2. 检查文件是否存在
if [ ! -f "$FILE1" ] || [ ! -f "$FILE2" ]; then
    echo "错误: 缺少必要的文件 ($FILE1 或 $FILE2)"
    exit 1
fi

echo "正在将 $FILE1 和 $FILE2 压缩并拆分..."

# 3. 压缩并拆分
# tar -cz: 同时将两个文件压缩
# split -b: 按大小拆分，-d 使用数字后缀
tar -cz "$FILE1" "$FILE2" | split -b $CHUNK_SIZE -d - "$PREFIX"

echo "拆分完成，生成的压缩分卷如下："
ls -lh ${PREFIX}*

# 4. 更新 .gitignore (防止原始大文件被上传)
for f in "$FILE1" "$FILE2"; do
    if ! grep -q "$f" .gitignore 2>/dev/null; then
        echo "$f" >> .gitignore
        echo "已将 $f 添加到 .gitignore"
    fi
done

# 5. Git 操作
echo "正在推送到 GitHub..."
git add "${PREFIX}*" .gitignore
git commit -m "Add split compressed parts of movie_ner_bert model (onnx + data)"
git push

echo "全部完成！"