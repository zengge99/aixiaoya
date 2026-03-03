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
# 引入特定类
from transformers import BertTokenizer, BertForTokenClassification, logging
import warnings

# --- 1. 全局配置区 ---
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 硬件与训练参数
NUM_THREADS = 4        
BATCH_SIZE = 32        
LR = 5e-5              
EPOCHS = 10            
MAX_LEN = 128          

# === 模型选择 (用户指定) ===
# 华为官方 TinyBERT (通用版 4层)
# BERT_HF_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
BERT_HF_NAME = 'bert-base-chinese'
BERT_LOCAL_FOLDER = "save_path"

# 权重文件
MODEL_WEIGHTS_PATH = "movie_model.pth"
DATA_FILE_PATTERN = "train_data*.txt"

SEED = 42
# 0=Outside, 1=Begin, 2=Inside
NUM_LABELS = 3 

torch.set_num_threads(NUM_THREADS)

# --- 2. 辅助工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextUtils:
    @staticmethod
    def preprocess_for_bert(text):
        if not text: return ""
        # 1. 分隔符处理
        # text = re.sub(r'[._]', ' ', text)
        # 2. 白名单保留
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        cn_puncts = r'【】（）《》：'
        # 3. 替换非法字符
        # pattern = f'[^{chars}{puncts}{cn_puncts}\\s]'
        # text = re.sub(pattern, ' ', text)
        return text

    @staticmethod
    def cleanup_result_(text):
        if not text: return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def cleanup_result(text):
        if not text: 
            return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'[.\-_\[\]()/]', ' ', text)
        # text = re.sub(r'\s+', ' ', text)
        return text

# --- 3. 自动下载逻辑 ---
def get_bert_path():
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        return BERT_LOCAL_FOLDER
    else:
        print(f"⚠️  本地未检测到模型，正在从 {BERT_HF_NAME} 下载...")
        try:
            tokenizer = BertTokenizer.from_pretrained(BERT_HF_NAME)
            # 注意：这里下载基础模型时，先不加 num_labels，仅保存架构
            model = BertForTokenClassification.from_pretrained(BERT_HF_NAME, num_labels=NUM_LABELS)
            
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            
            print(f"✅ 模型已下载并保存。")
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("💡 提示: 请尝试设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            sys.exit(1)

# --- 4. 模型封装 (Wrapper) ---
class NERModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 直接加载 BertForTokenClassification
        # num_labels=3 (O, B-Movie, I-Movie)
        self.bert = BertForTokenClassification.from_pretrained(model_path, num_labels=NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        # return_dict=True 返回对象，我们只需要 logits
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return output.logits  # Shape: [batch, seq_len, 3]

# --- 5. 数据集 (BIO 标注) ---
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
            
            escaped = re.escape(target_name).replace(r'\ ', r'.*')
            if re.search(escaped, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_name = self.samples[idx]
        
        # 1. 找位置
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        
        start_char, end_char = -1, -1
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()

        # 2. Tokenize
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

        # 3. 生成 Labels (Long Tensor)
        # 0: O (Outside)
        # 1: B (Begin)
        # 2: I (Inside)
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        if start_char != -1:
            found_start = False # 标记是否找到了第一个token
            
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0 or start == end: 
                    labels[i] = -100 # PyTorch CrossEntropyLoss 默认忽略 -100
                    continue
                
                # 计算重叠
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                
                # 判定是否属于电影名
                if token_len > 0 and (overlap / token_len > 0.4):
                    if not found_start:
                        labels[i] = 1 # B-Movie (首个 Token)
                        found_start = True
                    else:
                        labels[i] = 2 # I-Movie (后续 Token)
                else:
                    labels[i] = 0 # O

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- 6. 训练流程 ---
def validate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, mask)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            val_loss += loss.item()
    return val_loss / len(loader) if len(loader) > 0 else float('inf')

def run_train(incremental=False):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bert_path = get_bert_path()
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # 1. 查找数据文件并排序
    data_files = sorted(glob.glob(DATA_FILE_PATTERN))
    if not data_files:
        print(f"❌ 未找到训练数据 {DATA_FILE_PATTERN}"); return
    
    print(f"模式: {'【增量训练】' if incremental else '【全量训练】'}")
    
    final_train_lines = []
    final_val_lines = []
    
    # 使用独立随机实例确保切分逻辑永远一致
    rng = random.Random(SEED)

    # 2. 遍历文件，按文件进行确定性切分
    for i, f_path in enumerate(data_files):
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if '#' in l]
        
        # 确定性打乱
        rng.shuffle(lines)
        
        # 划分界限 (95% 训练, 5% 验证)
        split_idx = int(len(lines) * 0.95)
        file_train_pool = lines[:split_idx]
        file_val_pool = lines[split_idx:]

        # 增量训练逻辑：第一个文件视为“基础文件”
        if incremental and i == 0:
            # 从基础文件的训练池中仅抽取 5% (可调)
            sample_size = max(1, int(len(file_train_pool) * 0.05))
            selected_train = file_train_pool[:sample_size]
            print(f" 📂 [基础文件] {os.path.basename(f_path)}: 采样 {len(selected_train)} 条进入训练集 (池大小: {len(file_train_pool)})")
        else:
            selected_train = file_train_pool
            print(f" 📂 [新增/全量] {os.path.basename(f_path)}: 使用全部 {len(selected_train)} 条数据")

        final_train_lines.extend(selected_train)
        final_val_lines.extend(file_val_pool)

    # 3. 构建 Dataset
    train_ds = MovieDataset(final_train_lines, tokenizer)
    val_ds = MovieDataset(final_val_lines, tokenizer)
    print(f"📊 最终规模 -> 训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(2, NUM_THREADS))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型与 Loss
    model = NERModel(bert_path).to(device)
    class_weights = torch.tensor([1.0, 10.0, 8.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_loss = float('inf')

    # 5. 计算基线损失 (Baseline Loss)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"🔄 检测到现有权重，正在计算基线损失...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        best_loss = validate_model(model, val_loader, criterion, device)
        print(f"📈 当前模型基线 Val Loss: {best_loss:.4f}")
    else:
        print("🆕 未检测到模型权重，将从头开始训练。")

    # 6. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, mask)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # 验证并保存
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        print(f"   └─ Current Val Loss: {avg_val_loss:.4f} (Best: {best_loss:.4f})")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"   ✨ 验证集 Loss 优化，模型已更新保存。")
        else:
            print(f"   ⏳ 本轮未超过最佳模型，跳过保存。")

# --- 7. 预测流程 (适配 BIO) ---
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
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(logits, dim=2)[0] 
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    
    # --- 1. 严格切分实体片段 ---
    candidate_entities = []
    current_entity = []
    
    for i, pred_class in enumerate(preds):
        start, end = offset_mapping[i]
        if start == end: continue 
        
        conf = probs[i][pred_class]
        
        if pred_class == 1:  # B 标签：强制结束上一个，开始一个新的
            if current_entity:
                candidate_entities.append(current_entity)
            current_entity = [{'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end], 'label': 'B'}]
            
        elif pred_class == 2:  # I 标签：接在当前实体后面
            if current_entity:
                current_entity.append({'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end], 'label': 'I'})
            else:
                # 如果第一个就是 I，当做 B 处理（增强鲁棒性）
                current_entity = [{'start': start, 'end': end, 'conf': conf, 'token': raw_path[start:end], 'label': 'I'}]
        
        else:  # O 标签：结束当前实体
            if current_entity:
                candidate_entities.append(current_entity)
                current_entity = []
    
    if current_entity:
        candidate_entities.append(current_entity)

    # --- 2. Debug 打印所有候选者及其分数 ---
    has_dbg = os.path.exists("dbg")
    if has_dbg:
        print(f"\nPATH: {raw_path}")
        print("-" * 40)
        # 顺便打印下逐个 token 的情况
        for i, pred_class in enumerate(preds):
            s, e = offset_mapping[i]
            if s == e: continue
            lbl = "O" if pred_class == 0 else ("B" if pred_class == 1 else "I")
            print(f"{raw_path[s:e]:<10} | {lbl} | {probs[i][pred_class]:.4f}")
        print("-" * 40)

    # --- 3. 比较各个实体，取平均置信度最高的 ---
    final_res = ""
    best_score = -1.0
    
    for cand in candidate_entities:
        # 提取文本
        c_start = cand[0]['start']
        c_end = cand[-1]['end']
        raw_extract = raw_path[c_start:c_end]
        cleaned_text = TextUtils.cleanup_result(raw_extract)
        
        # 计算该片段的平均分
        avg_conf = np.mean([item['conf'] for item in cand])
        
        if has_dbg:
            print(f"Candidate: {raw_extract:<20} | Score: {avg_conf:.4f}")

        if cleaned_text and avg_conf > best_score:
            best_score = avg_conf
            final_res = cleaned_text

    # --- 4. 输出 ---
    if final_res:
        if has_dbg: print(f"Final Win: {final_res}")
        print(f"{raw_path}#{final_res}")
    else:
        print(f"{raw_path}#")

def load_inference_components():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ 未找到权重 {MODEL_WEIGHTS_PATH}，请先训练。")
        return None, None
    
    bert_path = get_bert_path()
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = NERModel(bert_path)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

# --- 修改入口逻辑 ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            run_train(incremental=False)
        elif cmd == '--inc':
            run_train(incremental=True)
        else:
            model, tok = load_inference_components()
            if model:
                if os.path.isfile(cmd):
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    predict_single(cmd, model, tok)
    else:
        run_train(incremental=False)