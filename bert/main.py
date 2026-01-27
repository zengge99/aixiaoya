import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os
import re
import random
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModel, logging

# --- 1. 全局配置区 ---
# 屏蔽 HuggingFace 的啰嗦警告
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 硬件与训练参数
NUM_THREADS = 4        # CPU 线程限制
BATCH_SIZE = 32        # 批大小 (低配机器建议 32 或 16)
LR = 5e-5              # 微调学习率 (不宜过大)
EPOCHS = 10            # 训练轮数
MAX_LEN = 128          # 序列截断长度

# 模型选择: 华为 TinyBERT (4层, 312维, 原生支持中文)
BERT_HF_NAME = "huawei-noah/TinyBERT_Chinese_4L_312D"
# 本地保存基础模型的文件夹 (自动创建)
BERT_LOCAL_FOLDER = "bert_base_huawei"
# 训练好的权重文件
MODEL_WEIGHTS_PATH = "movie_model_huawei.pth"
# 训练数据匹配模式
DATA_FILE_PATTERN = "train_data*.txt"

SEED = 42
THRESHOLD = 0.5        # 预测阈值
DEBUG_MODE = False     # 默认关闭，通过创建 'dbg' 文件开启

torch.set_num_threads(NUM_THREADS)

# --- 2. 辅助工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextUtils:
    @staticmethod
    def preprocess_for_bert(text):
        """
        预处理逻辑：
        1. 保持字符串长度严格不变 (1对1替换)，确保 offset_mapping 准确。
        2. 将 . 和 _ 替换为空格 (消除分词干扰)。
        3. 保留有语义的标点 ([], (), -, :, 等)。
        4. 其他乱码符号替换为空格。
        """
        if not text: return ""
        
        # 1. 核心分隔符处理：点和下划线 -> 空格
        text = re.sub(r'[._]', ' ', text)
        
        # 2. 白名单：保留 中英文、数字、结构性标点
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        cn_puncts = r'【】（）《》：'
        
        # 3. 非白名单字符 -> 空格
        pattern = f'[^{chars}{puncts}{cn_puncts}\s]'
        text = re.sub(pattern, ' ', text)
        return text

    @staticmethod
    def cleanup_result(text):
        """结果清洗：去头去尾，合并空格"""
        if not text: return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'\s+', ' ', text)
        return text

# --- 3. 自动下载与加载逻辑 ---
def get_bert_path():
    """
    检查本地是否有基础模型，没有则下载。
    返回有效的本地路径。
    """
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        # 文件夹存在且不为空，直接返回
        return BERT_LOCAL_FOLDER
    else:
        print(f"⚠️  本地未检测到模型，正在从 {BERT_HF_NAME} 下载...")
        print("    (这只需执行一次，后续将离线运行)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME)
            model = AutoModel.from_pretrained(BERT_HF_NAME)
            
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            
            print(f"✅ 模型已保存至: {BERT_LOCAL_FOLDER}")
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)

# --- 4. 模型定义 ---
class BertExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 加载本地 BERT
        self.bert = AutoModel.from_pretrained(model_path)
        # 动态获取隐藏层维度 (华为TinyBERT是312, 英文TinyBERT是128)
        self.hidden_size = self.bert.config.hidden_size 
        
        # 简单的分类头：Hidden -> 1 (概率)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        probs = self.classifier(sequence_output).squeeze(-1)
        return probs

# --- 5. 数据集定义 ---
class MovieDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len=MAX_LEN):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            target_name = target_name.strip()
            if not target_name: continue
            
            # 宽容匹配：确保 target 能在 input 里找到（忽略大小写和中间符号）
            escaped = re.escape(target_name).replace(r'\ ', r'.*')
            if re.search(escaped, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_name = self.samples[idx]
        
        # 1. 定位目标位置
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()
        else:
            start_char, end_char = -1, -1

        # 2. 预处理文本
        text_for_bert = TextUtils.preprocess_for_bert(input_path)

        # 3. Tokenize 并获取 Offset Mapping
        encoding = self.tokenizer(
            text_for_bert,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)

        # 4. 生成标签 (Token Level)
        labels = torch.zeros(self.max_len, dtype=torch.float)
        
        if start_char != -1:
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0 or start == end: continue 
                
                # 计算 Token 与 目标字符区间 的重叠度
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                
                # 如果重叠超过 40%，标记为 1
                if token_len > 0 and (overlap / token_len > 0.4):
                    labels[i] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- 6. 损失函数 ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=6.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, mask):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # 对正样本(电影名)加权
        weights = targets * self.pos_weight + (1 - targets)
        loss = loss * weights * mask
        return loss.sum() / (mask.sum() + 1e-6)

# --- 7. 训练流程 ---
def run_train():
    set_seed(SEED)
    
    # 准备基础模型
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # 读取数据
    data_files = glob.glob(DATA_FILE_PATTERN)
    if not data_files:
        print(f"❌ 未找到训练数据文件 ({DATA_FILE_PATTERN})"); return
        
    all_lines = []
    for f in data_files:
        with open(f, 'r', encoding='utf-8') as file:
            all_lines.extend(file.readlines())
    
    random.shuffle(all_lines)
    split = int(len(all_lines) * 0.95)
    train_ds = MovieDataset(all_lines[:split], tokenizer)
    val_ds = MovieDataset(all_lines[split:], tokenizer)
    
    print(f"数据准备完毕: 训练集 {len(train_ds)} 条 | 验证集 {len(val_ds)} 条")
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(2, NUM_THREADS))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = BertExtractor(bert_path)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"🔄 加载已有权重: {MODEL_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    else:
        print("🆕 开始从头训练...")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = WeightedBCELoss(pos_weight=6.0)
    best_loss = float('inf')
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for batch in pbar:
                optimizer.zero_grad()
                preds = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(preds, batch['labels'], batch['attention_mask'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    preds = model(batch['input_ids'], batch['attention_mask'])
                    loss = criterion(preds, batch['labels'], batch['attention_mask'])
                    val_loss += loss.item()
            
            avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            print(f"   └─ Val Loss: {avg_val:.4f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
                print(f"   ✨ 模型已优化并保存")
                
    except KeyboardInterrupt:
        print("\n🛑 用户停止训练")

# --- 8. 预测流程 (含 Debug 表格) ---
def predict_single(raw_path, model, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    # 1. 预处理
    text_for_bert = TextUtils.preprocess_for_bert(raw_path)
    inputs = tokenizer(
        text_for_bert,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN
    )
    
    # 2. 推理
    with torch.no_grad():
        probs = model(inputs['input_ids'], inputs['attention_mask'])[0].numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    
    # 3. 映射回字符级分数
    char_scores = np.zeros(len(raw_path))
    char_mask = np.zeros(len(raw_path), dtype=bool)
    
    for i, prob in enumerate(probs):
        start, end = offset_mapping[i]
        if start == end: continue # 跳过 [CLS]
        if end > len(raw_path): continue
        
        # 赋值分数
        char_scores[start:end] = prob
        if prob > THRESHOLD:
            char_mask[start:end] = True

    # --- Debug 打印 ---
    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"RAW : {raw_path}")
        print(f"BERT: {text_for_bert}")
        print("-" * 60)
        print(f"{'Idx':<4} | {'Char':<4} | {'Score':<8} | {'Status'}")
        print("-" * 60)
        for i, char in enumerate(raw_path):
            score = char_scores[i]
            # 仅显示非零分或可打印字符
            status = "✅" if score > THRESHOLD else "  "
            display_char = char if char.isprintable() else '?'
            print(f"{i:<4} | {display_char:<4} | {score:.4f}   | {status}")
        print(f"{'='*60}\n")
    # ----------------

    # 4. 后处理 (连通区域合并)
    final_res = ""
    true_indices = np.where(char_mask)[0]
    
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        for i in range(1, len(true_indices)):
            prev, curr = true_indices[i-1], true_indices[i]
            gap_str = raw_path[prev+1:curr]
            
            # Gap Filling: 如果中间断开的是标点/空格，且距离短，则视为一体
            if (curr - prev) < 5 and not any(c.isalnum() for c in gap_str):
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        # 取最长的片段
        best_grp = max(groups, key=len)
        start_idx = best_grp[0]
        end_idx = best_grp[-1] + 1
        
        raw_extract = raw_path[start_idx:end_idx]
        final_res = TextUtils.cleanup_result(raw_extract)

    if final_res:
        print(f"{raw_path}#{final_res}")
    else:
        print(f"{raw_path}#")

def load_inference_components():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ 未找到权重文件 {MODEL_WEIGHTS_PATH}，请先训练。")
        return None, None
    
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertExtractor(bert_path)
    
    # 这里的 strict=False 是为了兼容部分不需要的头，但这里架构一致，通常没问题
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

# --- 9. 主程序入口 ---
if __name__ == "__main__":
    # 检测 debug 文件
    if os.path.exists("dbg"):
        DEBUG_MODE = True
        print("🐞 调试模式已开启 (输出字符评分详情)")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'train':
            run_train()
        else:
            # 预测模式
            model, tok = load_inference_components()
            if model:
                if os.path.isfile(cmd):
                    # 文件批量预测
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    # 单条字符串预测
                    predict_single(cmd, model, tok)
    else:
        # 默认行为
        run_train()