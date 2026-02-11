#!/bin/bash
set -e  # 出错立即退出

# ===================== 极简配置项 =====================
TARGET_FILE="movie_ner_bert.onnx"  # 要压缩的目标文件
MAX_SIZE="50M"                     # 单个zip包最大大小（固定50M）
OUTPUT_PREFIX="movie_ner_bert"     # 压缩包前缀名

# ===================== 第一步：检查文件是否存在 =====================
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 错误：文件 $TARGET_FILE 不存在！"
    exit 1
fi

# 显示原始文件大小
echo "📄 原始文件：$TARGET_FILE | 大小：$(du -h $TARGET_FILE | awk '{print $1}')"

# ===================== 第二步：清理旧分片（避免残留） =====================
rm -f ${OUTPUT_PREFIX}_*.zip
rm -f checksum.txt
echo -e "\n🗑️  已清理旧的压缩分片"

# ===================== 第三步：拆分压缩成 ≤50M 的 zip 分片 =====================
echo -e "🚀 开始拆分压缩（单个包≤$MAX_SIZE）..."
# zip -s：指定分片大小；-q：静默模式；-m：压缩后删除原文件（可选，注释则保留原文件）
zip -q -s $MAX_SIZE ${OUTPUT_PREFIX}.zip $TARGET_FILE

# 重命名分片为友好格式（movie_ner_bert_00.zip、01.zip...）
# 1. 重命名主文件
mv ${OUTPUT_PREFIX}.zip ${OUTPUT_PREFIX}_00.zip
echo "✅ 生成：${OUTPUT_PREFIX}_00.zip"

# 2. 重命名子分片（.z01/.z02 → _01.zip/_02.zip）
for file in ${OUTPUT_PREFIX}.*.z*; do
    if [ -f "$file" ]; then
        # 提取分片序号（如 .z01 → 01）
        seq_num=$(printf "%02d" ${file##*.z})
        new_name="${OUTPUT_PREFIX}_${seq_num}.zip"
        mv "$file" "$new_name"
        echo "✅ 生成：$new_name"
    fi
done

# ===================== 第四步：生成校验文件（可选，验证完整性） =====================
echo -e "\n🔍 生成MD5校验文件（验证解压完整性）..."
md5sum $TARGET_FILE > checksum.txt
echo "✅ 生成：checksum.txt"

# ===================== 提示 git 推送命令 =====================
echo -e "\n🎉 拆分压缩完成！可执行以下命令推送到GitHub："
echo "git add ${OUTPUT_PREFIX}_*.zip checksum.txt"
echo "git commit -m \"Add movie_ner_bert.onnx split zip (≤50M)\""
echo "git push origin main"  # 分支名替换为你的（如 master）

# ===================== 解压还原提示（给使用者） =====================
echo -e "\n📝 解压还原方法："
echo "1. 下载所有分片到同一目录"
echo "2. 执行：zip -F ${OUTPUT_PREFIX}_00.zip --out ${OUTPUT_PREFIX}_merged.zip"
echo "3. 执行：unzip ${OUTPUT_PREFIX}_merged.zip"
echo "4. 验证完整性：md5sum -c checksum.txt"