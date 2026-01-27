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
import warnings

# --- 1. 全局配置区 ---
# 屏蔽警告
logging.set_verbosity_error()
warnings.filterwarnings("ignore") # 屏蔽 SyntaxWarning 等
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 硬件与训练参数
NUM_THREADS = 4        # CPU 线程限制
BATCH_SIZE = 32        # 批大小
LR = 5e-5              # 学习率
EPOCHS = 10            # 训练轮数
MAX_LEN = 128          # 序列长度

# 模型选择: UER TinyBERT (4层, 312维, 中文支持极好, 下载更稳)
# 架构与华为的一模一样，可以直接替换
BERT_HF_NAME = "uer/tinybert-base-chinese-4l-312d"

# 本地保存文件夹
BERT_LOCAL_FOLDER = "bert_base_chinese_tiny"
# 权重文件
MODEL_WEIGHTS_PATH = "movie_model_tinybert.pth"
# 数据匹配模式
DATA_FILE_PATTERN = "train_data*.txt"

SEED = 42
THRESHOLD = 0.5        
DEBUG_MODE = False     

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
        预处理：
        1. 保持长度不变。
        2. 将 . 和 _ 替换为空格。
        3. 保留中英文、数字、关键标点。
        """
        if not text: return ""
        
        # 1. 分隔符处理
        text = re.sub(r'[._]', ' ', text)
        
        # 2. 白名单：保留 中英文、数字
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        # 结构性标点
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        # 中文标点
        cn_puncts = r'【】（）《》：'
        
        # 3. 非白名单字符 -> 空格
        # [修复] 这里使用 \\s 避免 invalid escape sequence 警告
        pattern = f'[^{chars}{puncts}{cn_puncts}\\s]'
        text = re.sub(pattern, ' ', text)
        return text

    @staticmethod
    def cleanup_result(text):
        if not text: return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'\s+', ' ', text)
        return text

# --- 3. 自动下载逻辑 ---
def get_bert_path():
    """
    检查本地模型，若无则下载
    """
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        return BERT_LOCAL_FOLDER
    else:
        print(f"⚠️  本地未检测到模型，正在从 {BERT_HF_NAME} 下载...")
        print(f"    目标文件夹: {BERT_LOCAL_FOLDER}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME)
            model = AutoModel.from_pretrained(BERT_HF_NAME)
            
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            
            print(f"✅ 模型已下载并保存。")
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("\n💡 提示: 如果你在国内服务器，请先设置镜像环境变量再运行：")
            print("   export HF_ENDPOINT=https://hf-mirror.com")
            print("   python main.py train\n")
            sys.exit(1)

# --- 4. 模型定义 ---
class BertExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size 
        
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

# --- 5. 数据集 ---
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
            
            # 宽容匹配检查
            escaped = re.escape(target_name).replace(r'\ ', r'.*')
            if re.search(escaped, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_name = self.samples[idx]
        
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()
        else:
            start_char, end_char = -1, -1

        text_for_bert = TextUtils.preprocess_for_bert(input_path)

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

        labels = torch.zeros(self.max_len, dtype=torch.float)
        
        if start_char != -1:
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0 or start == end: continue 
                
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                
                if token_len > 0 and (overlap / token_len > 0.4):
                    labels[i] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- 6. Loss ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=6.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, mask):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weights = targets * self.pos_weight + (1 - targets)
        loss = loss * weights * mask
        return loss.sum() / (mask.sum() + 1e-6)

# --- 7. 训练流程 ---
def run_train():
    set_seed(SEED)
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    data_files = glob.glob(DATA_FILE_PATTERN)
    if not data_files:
        print(f"❌ 未找到训练数据 {DATA_FILE_PATTERN}"); return
        
    all_lines = []
    for f in data_files:
        with open(f, 'r', encoding='utf-8') as file:
            all_lines.extend(file.readlines())
    
    random.shuffle(all_lines)
    split = int(len(all_lines) * 0.95)
    train_ds = MovieDataset(all_lines[:split], tokenizer)
    val_ds = MovieDataset(all_lines[split:], tokenizer)
    
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(2, NUM_THREADS))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BertExtractor(bert_path)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"🔄 加载已有权重: {MODEL_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    
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
                print(f"   ✨ 模型已保存")
                
    except KeyboardInterrupt:
        print("\n🛑 训练停止")

# --- 8. 预测流程 ---
def predict_single(raw_path, model, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    text_for_bert = TextUtils.preprocess_for_bert(raw_path)
    inputs = tokenizer(
        text_for_bert,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN
    )
    
    with torch.no_grad():
        probs = model(inputs['input_ids'], inputs['attention_mask'])[0].numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    
    char_scores = np.zeros(len(raw_path))
    char_mask = np.zeros(len(raw_path), dtype=bool)
    
    for i, prob in enumerate(probs):
        start, end = offset_mapping[i]
        if start == end: continue 
        if end > len(raw_path): continue
        
        char_scores[start:end] = prob
        if prob > THRESHOLD:
            char_mask[start:end] = True

    # Debug 打印
    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"RAW : {raw_path}")
        print("-" * 60)
        print(f"{'Idx':<4} | {'Char':<4} | {'Score':<8} | {'Status'}")
        print("-" * 60)
        for i, char in enumerate(raw_path):
            score = char_scores[i]
            status = "✅" if score > THRESHOLD else "  "
            display_char = char if char.isprintable() else '?'
            print(f"{i:<4} | {display_char:<4} | {score:.4f}   | {status}")
        print(f"{'='*60}\n")

    # 后处理合并
    final_res = ""
    true_indices = np.where(char_mask)[0]
    
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        for i in range(1, len(true_indices)):
            prev, curr = true_indices[i-1], true_indices[i]
            gap_str = raw_path[prev+1:curr]
            
            # Gap Filling: 允许中间有少量标点断开
            if (curr - prev) < 5 and not any(c.isalnum() for c in gap_str):
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        # 提取最长片段
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
        print(f"❌ 未找到权重 {MODEL_WEIGHTS_PATH}，请先训练。")
        return None, None
    
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertExtractor(bert_path)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

# --- 9. 入口 ---
if __name__ == "__main__":
    if os.path.exists("dbg"):
        DEBUG_MODE = True
        print("🐞 调试模式开启")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            run_train()
        else:
            model, tok = load_inference_components()
            if model:
                if os.path.isfile(cmd):
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    predict_single(cmd, model, tok)
    else:
        run_train()