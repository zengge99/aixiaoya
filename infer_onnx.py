import onnxruntime as ort
import pickle
import re
import sys
import os
import string

# --- 全局配置（和 main.py 一致） ---
MAX_LEN = 300
THRESHOLD = 0.35
VOCAB_PATH = "vocab.pkl"
ONNX_MODEL_PATH = "movie_extractor.onnx"

# --- 必需工具类 TextUtils（从 main.py 复制） ---
class TextUtils:
    CN_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    @staticmethod
    def number2text(text):
        if not text: return text
        text = text.lstrip('0')
        if not text: return "零" 

        try:
            num = int(text)
        except ValueError:
            return text

        if num <= 10:
            return TextUtils.CN_NUMS[num]
        elif num < 20:
            return "十" + TextUtils.CN_NUMS[num % 10]
        elif num % 10 == 0:
            return TextUtils.CN_NUMS[num // 10] + "十"
        else:
            return TextUtils.CN_NUMS[num // 10] + "十" + TextUtils.CN_NUMS[num % 10]

    @staticmethod
    def fix_name(path, ai_result):
        # 新增：全英文直接返回
        if ai_result and all(ord(c) < 128 for c in ai_result):
            return ai_result

        replace_patterns = [
            r'Season\s*(\d{1,2})',              
            r'SE(\d{1,2})',                     
            r'(?<![a-zA-Z])S(\d{1,2})(?![a-zA-Z])', 
            r'第(\d{1,2})季'                    
        ]

        processed_result = ai_result
        replaced_flag = False 

        def replace_func(match):
            nonlocal replaced_flag
            replaced_flag = True
            num = match.group(1)
            cn_num = TextUtils.number2text(num)
            return f" 第{cn_num}季 " 

        for pattern in replace_patterns:
            if re.search(pattern, processed_result, re.IGNORECASE):
                processed_result = re.sub(pattern, replace_func, processed_result, flags=re.IGNORECASE)
        
        processed_result = re.sub(r'\s+', ' ', processed_result).strip()

        if replaced_flag:
            return processed_result

        path_search_patterns = [
            r'Season\s*(\d{1,2})',
            r'SE(\d{1,2})',
            r'第(\d{1,2})季',
            r'(?<![A-Za-z])S(\d{1,2})'
        ]

        for pattern in path_search_patterns:
            match = re.search(pattern, path, re.IGNORECASE)
            if match:
                num = match.group(1)
                cn_num = TextUtils.number2text(num)
                suffix = f"第{cn_num}季"
                if suffix not in processed_result:
                    return f"{processed_result} {suffix}"
                break 
        
        return processed_result

# --- ONNX 初始化函数（仅执行1次） ---
def init_onnx_session():
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print(f"❌ 缺失文件：需 {ONNX_MODEL_PATH} 和 {VOCAB_PATH} 在同目录")
        return None, None
    
    # 加载词表
    with open(VOCAB_PATH, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    # 加载 ONNX 模型（CPU 推理，无 GPU 依赖）
    sess = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions()
    )
    return sess, char_to_idx

# --- 单条路径预测函数 ---
def predict_single_path(path, sess, char_to_idx):
    if '#' in path:
        print(path)
        return
    
    # 输入预处理
    input_ids = [char_to_idx.get(c.lower(), 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    padded = [padded]  # 构造 batch 维度 [1, MAX_LEN]

    # ONNX Runtime 推理（无梯度，速度快）
    outputs = sess.run(["probs"], {"input_ids": padded})
    probs = outputs[0][0][:len(path)]  # 截断到原始路径长度

    # --- 后处理逻辑（和 main.py 完全一致） ---
    selected_mask = [False] * len(probs)
    
    # 1. 阈值筛选
    for i, p in enumerate(probs):
        if p > THRESHOLD:
            selected_mask[i] = True
            
    # 2. 空洞填补
    gap_limit = 2 
    for i in range(len(probs)):
        if selected_mask[i]:
            for j in range(i + 1, min(i + gap_limit + 2, len(probs))):
                if selected_mask[j]:
                    for k in range(i + 1, j):
                        if path[k] not in ['/', '\\']:
                            selected_mask[k] = True
                    break

    # 结果拼接
    res_list = [path[i] for i, is_sel in enumerate(selected_mask) if is_sel]
    raw_result = "".join(res_list)
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    clean_result = re.sub(r'\s+', ' ', clean_result)
    clean_result = clean_result.strip("/()# “”.-")

    # 验证连续性
    if clean_result:
        escaped_clean = re.escape(clean_result)
        verify_pattern = escaped_clean.replace(r'\ ', r'[._\s\-\(\)\[\]]*')
        if not re.search(verify_pattern, path, re.IGNORECASE):
            clean_result = ""

    # 混合模式修复
    if clean_result:
        clean_result = TextUtils.fix_name(path, clean_result)

    # 输出结果
    print(f"{path}#{clean_result}" if clean_result else f"{path}#")

# --- 批量预测函数 ---
def run_batch_predict(file_path):
    sess, char_to_idx = init_onnx_session()
    if not sess:
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    total_lines = len(lines)
    print(f"开始批量处理 {total_lines} 条路径...")
    for idx, line in enumerate(lines):
        if idx % 100 == 0 and idx > 0:
            print(f"进度: {idx}/{total_lines}")
        predict_single_path(line, sess, char_to_idx)
    print(f"✅ 批量处理完成！")

# --- 入口控制 ---
if __name__ == "__main__":
    # 调试模式：同目录存在 dbg 文件则开启
    DEBUG_MODE = os.path.exists("dbg")
    if DEBUG_MODE:
        print("🔍 调试模式已开启")

    if len(sys.argv) > 1:
        input_arg = sys.argv[1]
        # 批量预测：输入是文件路径
        if os.path.exists(input_arg) and os.path.isfile(input_arg):
            run_batch_predict(input_arg)
        # 单条预测：输入是路径字符串
        else:
            sess, char_to_idx = init_onnx_session()
            if sess:
                predict_single_path(input_arg, sess, char_to_idx)
    else:
        print("用法:")
        print("  单条预测: python infer_onnx.py \"你的文件路径\"")
        print("  批量预测: python infer_onnx.py 路径文件.txt")
