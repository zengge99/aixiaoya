#!/bin/bash

# 1. 配置变量
FILE1="movie_ner_bert.onnx"
FILE2="movie_ner_bert.onnx.data"
DIR1="save_path"
TEMP_ZIP="movie_model_full.zip"
CHUNK_SIZE="49M"
PREFIX="movie_model_full.zip.part"

# 2. 检查文件和目录是否存在
echo "检查文件和文件夹..."
if [ ! -f "$FILE1" ]; then echo "错误: 找不到 $FILE1"; exit 1; fi
if [ ! -f "$FILE2" ]; then echo "错误: 找不到 $FILE2"; exit 1; fi
if [ ! -d "$DIR1" ]; then echo "错误: 找不到目录 $DIR1"; exit 1; fi

# 3. 创建 ZIP 压缩包
# -r: 递归压缩文件夹
# -q: 静默模式
echo "正在打包压缩 $FILE1, $FILE2 和 $DIR1 ..."
zip -rq "$TEMP_ZIP" "$FILE1" "$FILE2" "$DIR1"

# 4. 拆分压缩包
echo "正在拆分压缩包为 ${CHUNK_SIZE} 的分卷..."
split -b $CHUNK_SIZE -d "$TEMP_ZIP" "$PREFIX"

# 5. 清理临时的巨大压缩包
rm "$TEMP_ZIP"

echo "拆分完成，分卷如下："
ls -lh ${PREFIX}*

# 6. 自动更新 .gitignore
echo "更新 .gitignore..."
for item in "$FILE1" "$FILE2" "$DIR1" "$TEMP_ZIP"; do
    if ! grep -q "^$item$" .gitignore 2>/dev/null; then
        echo "$item" >> .gitignore
    fi
done

# 7. Git 推送
echo "正在推送到 GitHub..."
git add "${PREFIX}*" .gitignore
git commit -m "Add split zip parts of model and save_path directory"
git push

echo "全部任务已完成！"