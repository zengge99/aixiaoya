import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import sys
import os
import re
import random
import numpy as np
import glob  # 引入glob用于匹配多个文件

# --- 全局核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 64
LR = 1e-4            # 学习率
EPOCHS = 50          # 训练轮数
MAX_LEN = 150        # 最大路径长度
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
# 数据文件匹配模式 (匹配 train_data.txt, train_data_2.txt 等)
DATA_FILE_PATTERN = "train_data*.txt" 
SEED = 42            # 🎲 固定随机种子

# --- 预测/调试配置 ---
DEBUG_MODE = True    # 开启后显示全路径所有字符得分
THRESHOLD = 0.2      # 核心判定阈值
SMOOTH_VAL = 0.05    # 辅助判定阈值（用于救回中间字符）

# 必须在 import torch 之后立即设置
torch.set_num_threads(NUM_THREADS)

# --- 🛠️ 辅助函数：固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证cudnn可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 移植的 JS 逻辑工具类 ---
class TextUtils:
    CN_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    @staticmethod
    def number2text(text):
        """
        将数字字符串转换为中文数字
        """
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
        """
        AI 结果后处理：修正或补全季数信息
        """
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

        # 1. 尝试在 AI 结果内部直接替换
        for pattern in replace_patterns:
            if re.search(pattern, processed_result, re.IGNORECASE):
                processed_result = re.sub(pattern, replace_func, processed_result, flags=re.IGNORECASE)
        
        processed_result = re.sub(r'\s+', ' ', processed_result).strip()

        if replaced_flag:
            return processed_result

        # 2. (兜底) 从原路径找季数追加
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

# --- 模型结构定义 ---
class FilmExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        return self.fc(gru_out).squeeze(-1)

# --- 数据集定义 ---
class MovieDataset(Dataset):
    def __init__(self, lines, char_to_idx, max_len=MAX_LEN):
        self.samples = []
        skipped_count = 0
        
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            target_name = target_name.strip()
            
            escaped_target = re.escape(target_name)
            pattern = escaped_target.replace(r'\ ', r'[._\s]+')
            match = re.search(pattern, input_path, re.IGNORECASE)
            
            if match:
                start_idx = match.start()
                end_idx = match.end()
                
                input_ids = [char_to_idx.get(c, 1) for c in input_path[:max_len]]
                labels = [0.0] * len(input_ids)
                
                limit = min(end_idx, max_len)
                for i in range(start_idx, limit):
                    labels[i] = 1.0
                
                pad_len = max_len - len(input_ids)
                self.samples.append((
                    torch.tensor(input_ids + [0] * pad_len), 
                    torch.tensor(labels + [0.0] * pad_len)
                ))
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"Dataset Info: 跳过了 {skipped_count} 条无法匹配标签的数据。")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 🛠️ 辅助函数：验证集计算 ---
def validate_one_epoch(model, loader, criterion):
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for vx, vy in loader:
            pred = model(vx)
            loss = criterion(pred, vy)
            v_loss += loss.item()
    return v_loss / len(loader) if len(loader) > 0 else 0

# --- 训练逻辑 ---
def run_train(incremental=False):
    # 设置全局种子
    set_seed(SEED)
    mode_str = "【增量训练模式】" if incremental else "【全量训练模式】"
    print(f"{mode_str} 随机种子已固定为: {SEED}")

    # 1. 搜索所有匹配的文件
    data_files = glob.glob(DATA_FILE_PATTERN)
    # 排序以保证每次运行读取顺序一致，确保 index 0 总是同一个文件
    data_files.sort()
    
    if not data_files:
        print(f"❌ 未找到匹配 {DATA_FILE_PATTERN} 的数据文件。"); return
    
    print(f"发现 {len(data_files)} 个数据文件: {data_files}")

    all_train_lines = []
    all_val_lines = []
    
    # 2. 遍历每个文件
    # 使用独立的 Random 实例进行 shuffle，不影响全局状态
    rng = random.Random(SEED)
    
    for i, f_path in enumerate(data_files):
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if '#' in l.strip()]
        
        # 确定性打乱
        rng.shuffle(lines)
        
        total_raw = len(lines)
        if total_raw == 0: continue
        
        # --- 增量训练核心逻辑 ---
        if incremental and i == 0:
            # 如果是增量模式，且是第一个文件（旧数据），只保留 10%
            keep_count = int(total_raw * 0.1)
            # 至少保留1条，避免空列表
            if keep_count == 0 and total_raw > 0: keep_count = 1
            
            lines = lines[:keep_count]
            print(f"  └─ [Old Data] {os.path.basename(f_path)}: 仅取 10% ({keep_count}/{total_raw}条)")
        else:
            # 其他情况（全量模式 或 增量模式下的新文件），保留 100%
            print(f"  └─ [New Data] {os.path.basename(f_path)}: 读取全量 ({total_raw}条)")

        # 3. 对筛选后的数据进行 训练/验证 切分 (90% / 10%)
        # 即使是 Old Data，我们也切分出验证集，以保证验证 Loss 的有效性
        current_total = len(lines)
        train_count = int(current_total * 0.9)
        if train_count == 0 and current_total > 0: train_count = current_total
        
        train_part = lines[:train_count]
        val_part = lines[train_count:]
        
        all_train_lines.extend(train_part)
        all_val_lines.extend(val_part)

    print(f"\n数据集准备完毕: 训练集 {len(all_train_lines)} 条 | 验证集 {len(all_val_lines)} 条")

    # 4. 构建或加载词表 (基于所有数据)
    all_lines_for_vocab = all_train_lines + all_val_lines
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
        print("已加载现有词表。")
    else:
        # 注意：如果是增量训练且没有旧词表，可能会漏掉旧数据里被丢弃的那90%字符
        # 但通常增量训练意味着已经有模型和词表了。
        raw_paths = [l.split('#')[0] for l in all_lines_for_vocab]
        all_chars = set("".join(raw_paths))
        char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}
        char_to_idx['<PAD>'], char_to_idx['<UNK>'] = 0, 1
        with open(VOCAB_PATH, 'wb') as f: pickle.dump(char_to_idx, f)
        print(f"已创建新词表，包含 {len(char_to_idx)} 个字符。")

    # 5. 创建 Dataset 和 DataLoader
    train_ds = MovieDataset(all_train_lines, char_to_idx)
    val_ds = MovieDataset(all_val_lines, char_to_idx)

    if len(train_ds) < 1:
        print("有效样本数量不足，无法进行训练。"); return

    # 这里使用了 generator 来确保 shuffle 的完全可复现性
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FilmExtractor(len(char_to_idx))
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')

    # 加载模型逻辑
    if os.path.exists(MODEL_PATH):
        print(f"检测到现有模型，加载权重以 LR={LR} 继续微调...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
        if len(val_ds) > 0:
            print("正在计算当前模型的初始验证集 Loss (基准线)...")
            initial_val_loss = validate_one_epoch(model, val_loader, criterion)
            best_val_loss = initial_val_loss 
            print(f"当前模型基准 Loss: {best_val_loss:.4f}")
    else:
        print("🆕 未检测到模型，将从头开始训练。")
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}")
            for x, y in pbar:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if len(val_ds) > 0:
                avg_val_loss = validate_one_epoch(model, val_loader, criterion)
                
                if avg_val_loss < best_val_loss:
                    print(f" ✨ Loss 优化 ({best_val_loss:.4f} -> {avg_val_loss:.4f})，模型已更新。")
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), MODEL_PATH)
                else:
                    print(f" ⏳ 验证集 Loss: {avg_val_loss:.4f} (未提升，最佳: {best_val_loss:.4f})")
            else:
                torch.save(model.state_dict(), MODEL_PATH)
                print(" ⚠️ 无验证集，模型已保存。")
                
    except KeyboardInterrupt: print("\n🛑 用户手动停止训练。")

# --- 预测逻辑 ---
def run_predict(path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("错误: 找不到模型或词表文件。请先运行训练。"); return

    with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
    model = FilmExtractor(len(char_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    input_ids = [char_to_idx.get(c, 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    
    with torch.no_grad():
        probs = model(torch.tensor([padded]))[0][:len(path)].numpy()

    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")

    res_list = []
    for i, p in enumerate(probs):
        is_high = p > THRESHOLD
        is_bridge = False
        if not is_high and p > SMOOTH_VAL:
            left_high = probs[i-1] > THRESHOLD if i > 0 else False
            right_high = probs[i+1] > THRESHOLD if i < len(probs)-1 else False
            if left_high and right_high:
                is_bridge = True
        
        if is_high or is_bridge:
            res_list.append(path[i])
    
    raw_result = "".join(res_list)
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    clean_result = re.sub(r'\s+', ' ', clean_result)
    clean_result = clean_result.strip("/()# “”.-")

    # 1. 验证连续性
    if clean_result:
        escaped_clean = re.escape(clean_result)
        verify_pattern = escaped_clean.replace(r'\ ', r'[._\s\-\(\)\[\]]*')
        if not re.search(verify_pattern, path, re.IGNORECASE):
            if DEBUG_MODE:
                print(f"[验证失败] '{clean_result}' 无法在原路径中连续匹配，判定为无效提取。")
            clean_result = ""

    # 2. 混合模式修复
    if clean_result:
        clean_result = TextUtils.fix_name(path, clean_result) 

    if DEBUG_MODE: 
        print(f"提取原文: {raw_result}")
        print(f"最终结果: {clean_result}\n")
    else: 
        print(clean_result)

# --- 入口控制 ---
if __name__ == "__main__":
    # 如果有参数
    if len(sys.argv) > 1:
        # 检查是否为增量训练标记
        if sys.argv[1] == '--inc':
            run_train(incremental=True)
        else:
            # 否则视为预测路径
            run_predict(sys.argv[1])
    else:
        # 默认全量训练
        run_train(incremental=False)