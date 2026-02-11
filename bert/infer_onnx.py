import onnxruntime as ort
from transformers import BertTokenizer
import pickle
import re
import sys
import os
import argparse
import numpy as np
from flask import Flask, request, jsonify

# --- 全局配置 ---
MAX_LEN = 128
ONNX_MODEL_PATH = "movie_ner_bert.onnx"
DEBUG_MODE = os.path.exists("dbg")

# --- 必需工具类 TextUtils ---
class TextUtils:
    CN_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    @staticmethod
    def cleanup_result(text):
        if not text: 
            return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'[.\-_\[\]()/]', ' ', text)
        # text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def cn_to_arabic(cn_str):
        """将中文数字（一到九十九）转换为字符串格式的阿拉伯数字"""
        cn_num_map = {'零':0, '一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9}
        
        # 如果本身就是阿拉伯数字，直接返回
        if cn_str.isdigit():
            return cn_str
        
        # 处理逻辑
        if cn_str == "十":
            return "10"
        
        res = 0
        if "十" in cn_str:
            parts = cn_str.split("十")
            # 处理 "二十..." 或 "十..."
            # 如果 "十" 在开头（如十一），前部分为空
            prefix = parts[0]
            suffix = parts[1]
            
            if prefix: # 二十...
                res += cn_num_map[prefix] * 10
            else: # 十...
                res += 10
                
            if suffix: # ...十一
                res += cn_num_map[suffix]
        else:
            # 仅有个位数
            res = cn_num_map.get(cn_str, cn_str)
            
        return str(res)

    @staticmethod
    def simplify_season_name(text):
        """
        核心转换函数
        例如: '功夫熊猫 第十一季' -> '功夫熊猫11'
        """
        # 正则匹配：匹配“第”后面跟着的一串中文数字或阿拉伯数字，直到“季”
        pattern = r'\s*第([一二三四五六七八九十\d]+)季'
        
        def replace_func(match):
            cn_val = match.group(1)
            return TextUtils.cn_to_arabic(cn_val)

        # 使用 re.sub 进行替换
        result = re.sub(pattern, replace_func, text)
        return result.strip()

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
    def fix_name_internal(path, ai_result):
        if ai_result and all(ord(c) < 128 for c in ai_result):
            return ai_result
        processed_result = ai_result

        cn_season_pattern = r'第[一二三四五六七八九十]+季'
        cn_match = re.search(cn_season_pattern, path)
        if cn_match:
            suffix = cn_match.group(0)
            if suffix not in processed_result:
                return f"{processed_result} {suffix}".strip()
            return processed_result

        replaced_flag = False 

        replace_patterns = [
            r'Season\s*(\d{1,2})',              
            r'SE(\d{1,2})',                     
            r'(?<![a-zA-Z])S(\d{1,2})(?![a-zA-Z])', 
            r'第(\d{1,2})季'                    
        ]

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

    @staticmethod
    def fix_name(path, ai_result):
        result = TextUtils.fix_name_internal(path, ai_result).replace("第一季", "", 1).strip()
        # tmdb不太认“功夫熊猫 第三季”这种，要转换成“功夫熊猫3”
        return TextUtils.simplify_season_name(result)

def get_resource_path(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    if hasattr(sys, '_MEIPASS'):
        bundle_path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(bundle_path):
            return bundle_path
    exe_dir_path = os.path.join(os.path.dirname(sys.executable), relative_path)
    if os.path.exists(exe_dir_path):
        return exe_dir_path
    return relative_path

# --- ONNX 初始化 ---
def init_onnx_session():
    actual_onnx_path = get_resource_path(ONNX_MODEL_PATH)

    tokenizer = BertTokenizer.from_pretrained("save_path")
    sess = ort.InferenceSession(actual_onnx_path, providers=['CPUExecutionProvider'])
    
    return sess, tokenizer

def softmax_np(x):
    """
    使用 NumPy 实现 Softmax，处理 BERT 输出的 Logits。
    x 的形状通常是 (1, seq_len, 3) 或 (seq_len, 3)
    """
    # 减去最大值是为了数值稳定性，防止 np.exp 计算出无穷大 (Overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def do_inference(raw_path, ort_session, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    # --- 1. 前处理与 Tokenization ---
    text_for_bert = raw_path
    inputs = tokenizer(
        text_for_bert,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN,
        padding=False # ONNX 支持动态长度时建议不强制填充到 128 以提升速度
    )
    
    # 转换为 NumPy 格式并增加 Batch 维度 [1, seq_len]
    input_ids = np.array(inputs['input_ids'], dtype=np.int64)[None, :]
    attention_mask = np.array(inputs['attention_mask'], dtype=np.int64)[None, :]

    # --- 2. ONNX 执行推理 ---
    # ort_session.run(输出节点名列表, 输入字典)
    # 输入字典的 key 必须与导出时的 input_names ["input_ids", "attention_mask"] 一致
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    logits = ort_session.run(None, ort_inputs)[0] # 输出形状 [1, seq_len, 3]
    
    # --- 3. 后处理 (Logits -> Probs -> Preds) ---
    probs_seq = softmax_np(logits[0]) # 取第一个 batch 并转为概率 [seq_len, 3]
    preds = np.argmax(probs_seq, axis=1) # [seq_len]
    
    offset_mapping = inputs['offset_mapping']
    
    # --- 4. BIO 标签解析 (逻辑同原版) ---
    candidate_entities = []
    current_entity = []
    
    for i, pred_class in enumerate(preds):
        start, end = offset_mapping[i]
        if start == end: continue # 跳过 [CLS], [SEP] 或填充部分
        
        conf = probs_seq[i][pred_class]
        
        if pred_class == 1:  # B-Movie
            if current_entity:
                candidate_entities.append(current_entity)
            current_entity = [{'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end]}]
            
        elif pred_class == 2:  # I-Movie
            if current_entity:
                current_entity.append({'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end]})
            else:
                # 容错处理：若只有 I 出现，视作起始
                current_entity = [{'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end]}]
        
        else:  # O (Outside)
            if current_entity:
                candidate_entities.append(current_entity)
                current_entity = []
    
    if current_entity:
        candidate_entities.append(current_entity)

    # --- 5. 结果筛选 (逻辑同原版) ---
    has_dbg = os.path.exists("dbg")
    if has_dbg:
    print(f"\nPATH: {raw_path}")
    print("-" * 40)
    # 顺便打印下逐个 token 的情况
    for i, pred_class in enumerate(preds):
        s, e = offset_mapping[i]
        if s == e: continue
        lbl = "O" if pred_class == 0 else ("B" if pred_class == 1 else "I")
        print(f"{raw_path[s:e]:<10} | {lbl} | {probs_seq[i][pred_class]:.4f}")
    print("-" * 40)

    final_res = ""
    best_score = -1.0
    
    for cand in candidate_entities:
        c_start = cand[0]['start']
        c_end = cand[-1]['end']
        raw_extract = raw_path[c_start:c_end]
        cleaned_text = TextUtils.cleanup_result(raw_extract)
        
        avg_conf = np.mean([item['conf'] for item in cand])
        
        if cleaned_text and avg_conf > best_score:
            best_score = avg_conf
            final_res = cleaned_text

    if final_res:
        final_res = TextUtils.fix_name(raw_path, final_res)
    return final_res

# --- 预测动作封装 ---
def predict_single_path(path, sess, tokenizer):
    res = do_inference(path, sess, tokenizer)
    if res:
        print(f"{path}#{res}")
    else:
        print(f"{path}")

def run_batch_predict(file_path, sess, tokenizer):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    for line in lines:
        predict_single_path(line, sess, tokenizer)

# --- HTTP 服务模式 ---
def start_server(port):
    app = Flask(__name__)

    @app.route('/')
    def api_extract():
        q = request.args.get('q', '')
        if not q:
            return jsonify({"error": "missing parameter q"}), 400
        sess, tokenizer = init_onnx_session()
        result = do_inference(q, sess, tokenizer)
        print(f"{result}")
        return result  # 直接返回提取出的字符串

    print(f"🚀 HTTP 服务已启动: http://0.0.0.0:{port}")
    print(f"📌 使用示例: http://127.0.0.1:{port}/?q=你的影片路径")
    app.run(host='0.0.0.0', port=port, debug=False)

# --- 入口控制 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电影名称提取工具")
    parser.add_argument("input", nargs="?", help="影片路径字符串 或 路径列表文件(.txt)")
    parser.add_argument("--srv", type=int, help="启动 HTTP 服务模式，指定端口号")

    args = parser.parse_args()

    # 初始化模型
    sess, tokenizer = init_onnx_session()
    if not sess:
        sys.exit(1)

    # 优先判断是否启动服务
    if args.srv:
        start_server(args.srv)
    
    # 其次判断是否有输入路径进行单条或批量预测
    elif args.input:
        if os.path.exists(args.input) and os.path.isfile(args.input):
            run_batch_predict(args.input, sess, tokenizer)
        else:
            predict_single_path(args.input, sess, tokenizer)
    else:
        parser.print_help()