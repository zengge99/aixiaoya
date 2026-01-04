import onnxruntime as ort
import pickle
import re
import sys
import os
import argparse
from flask import Flask, request, jsonify

# --- 全局配置 ---
MAX_LEN = 300
THRESHOLD = 0.35
VOCAB_PATH = "vocab.pkl"
ONNX_MODEL_PATH = "movie_extractor.onnx"
DEBUG_MODE = os.path.exists("dbg")

# --- 必需工具类 TextUtils ---
class TextUtils:
    CN_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

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
    actual_vocab_path = get_resource_path(VOCAB_PATH)

    if not os.path.exists(actual_onnx_path) or not os.path.exists(actual_vocab_path):
        print(f"❌ 缺失文件：需 {ONNX_MODEL_PATH} 和 {VOCAB_PATH} 在同目录")
        return None, None
    
    with open(actual_vocab_path, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    sess = ort.InferenceSession(
        actual_onnx_path,
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions()
    )
    return sess, char_to_idx

# --- 核心推理逻辑提取 ---
def do_inference(path, sess, char_to_idx):
    if '#' in path:
        return "" # 原逻辑中碰到#号直接返回空或打印原路径
    
    input_ids = [char_to_idx.get(c.lower(), 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    padded = [padded]

    outputs = sess.run(["probs"], {"input_ids": padded})
    probs = outputs[0][0][:len(path)]
    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")
    selected_mask = [False] * len(probs)
    for i, p in enumerate(probs):
        if p > THRESHOLD:
            selected_mask[i] = True
            
    gap_limit = 2 
    for i in range(len(probs)):
        if selected_mask[i]:
            for j in range(i + 1, min(i + gap_limit + 2, len(probs))):
                if selected_mask[j]:
                    for k in range(i + 1, j):
                        if path[k] not in ['/', '\\']:
                            selected_mask[k] = True
                    break

    res_list = [path[i] for i, is_sel in enumerate(selected_mask) if is_sel]
    raw_result = "".join(res_list)
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    clean_result = re.sub(r'\s+', ' ', clean_result)
    clean_result = clean_result.strip("/()# “”.-")

    if clean_result:
        escaped_clean = re.escape(clean_result)
        verify_pattern = escaped_clean.replace(r'\ ', r'[._\s\-\(\)\[\]]*')
        if not re.search(verify_pattern, path, re.IGNORECASE):
            clean_result = ""

    if clean_result:
        clean_result = TextUtils.fix_name(path, clean_result)
    
    return clean_result

# --- 预测动作封装 ---
def predict_single_path(path, sess, char_to_idx):
    res = do_inference(path, sess, char_to_idx)
    if res:
        print(f"{path}#{res}")
    else:
        print(f"{path}")

def run_batch_predict(file_path, sess, char_to_idx):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    for line in lines:
        predict_single_path(line, sess, char_to_idx)

# --- HTTP 服务模式 ---
def start_server(port):
    app = Flask(__name__)

    @app.route('/')
    def api_extract():
        q = request.args.get('q', '')
        if not q:
            return jsonify({"error": "missing parameter q"}), 400
        sess, char_to_idx = init_onnx_session()
        result = do_inference(q, sess, char_to_idx)
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
    sess, char_to_idx = init_onnx_session()
    if not sess:
        sys.exit(1)

    # 优先判断是否启动服务
    if args.srv:
        start_server(args.srv)
    
    # 其次判断是否有输入路径进行单条或批量预测
    elif args.input:
        if os.path.exists(args.input) and os.path.isfile(args.input):
            run_batch_predict(args.input, sess, char_to_idx)
        else:
            predict_single_path(args.input, sess, char_to_idx)
    else:
        parser.print_help()