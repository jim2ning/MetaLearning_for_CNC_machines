import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import copy
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

def load_hyperparameters(config_path):
    """하이퍼파라미터 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

class LSTM_AE(nn.Module):
    """LSTM AutoEncoder"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_layers=2, dropout_rate=0.2):
        super(LSTM_AE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rate
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout_rate
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def encode(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, _) = self.encoder_lstm(x)
        # 마지막 timestep의 hidden state 사용
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        latent = self.encoder_fc(last_hidden)  # (batch_size, latent_dim)
        return latent
    
    def decode(self, latent, seq_len):
        # latent: (batch_size, latent_dim)
        batch_size = latent.size(0)
        
        # latent를 hidden state로 변환
        hidden = self.decoder_fc(latent)  # (batch_size, hidden_dim)
        
        # hidden state를 seq_len만큼 반복
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM 디코딩
        lstm_out, _ = self.decoder_lstm(hidden)
        output = self.output_layer(lstm_out)  # (batch_size, seq_len, input_dim)
        
        return output
    
    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed

class MAML_LSTM_AE:
    """개선된 MAML for LSTM AutoEncoder"""
    def __init__(self, model, meta_lr, inner_lr, inner_steps):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # 개선된 메타 옵티마이저
        self.meta_optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=meta_lr, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        # 학습률 스케줄러
        self.meta_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optimizer, 
            T_0=20,              # 주기를 더 길게
            T_mult=2,            # 주기가 2배씩 증가
            eta_min=1e-6  
        )
        
        # 그래디언트 축적을 위한 변수
        self.grad_accumulation_steps = 1
        self.current_step = 0
        
    def functional_forward(self, x, weights):
        """함수형 순전파 - 주어진 가중치로 모델 실행"""
        # LSTM AutoEncoder의 각 레이어별 가중치 적용
        # 이 부분은 모델 구조에 맞게 구현 필요
        return self.model(x)  # 임시로 원본 모델 사용
    
    def compute_inner_loss(self, pred, target, step=0):
        """내부 루프 손실 - 단순화된 버전"""
        base_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        return base_loss + 0.01 * l1_loss
    
    def meta_train_step(self, task_batch):
        """개선된 메타 학습 단계"""
        try:
            meta_loss = 0.0
            valid_tasks = 0
            
            for task in task_batch:
                support_x, _, query_x, _ = task
                
                # 디바이스 이동
                support_x = support_x.to(device)
                query_x = query_x.to(device)
                
                # 데이터 검증
                if torch.isnan(support_x).any() or torch.isnan(query_x).any():
                    continue
                if torch.isinf(support_x).any() or torch.isinf(query_x).any():
                    continue
                if support_x.shape[0] == 0 or query_x.shape[0] == 0:
                    continue
                
                # Inner loop adaptation
                self.model.train()
                temp_model = copy.deepcopy(self.model)
                temp_optimizer = optim.Adam(temp_model.parameters(), lr=self.inner_lr)
                
                # 다중 스텝 적응
                for step in range(self.inner_steps):
                    temp_optimizer.zero_grad()
                    support_pred = temp_model(support_x)
                    support_loss = self.compute_inner_loss(support_pred, support_x, step)
                    support_loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)
                    temp_optimizer.step()
                
                # Query loss with adapted model
                query_pred = temp_model(query_x)
                query_loss = F.mse_loss(query_pred, query_x)
                
                # 손실 검증
                if not torch.isnan(query_loss) and not torch.isinf(query_loss):
                    meta_loss += query_loss
                    valid_tasks += 1
                
                # 메모리 정리
                del temp_model, temp_optimizer
                torch.cuda.empty_cache()
            
            if valid_tasks == 0:
                return None
            
            # 평균 메타 손실
            meta_loss = meta_loss / valid_tasks
            
            # 그래디언트 축적
            meta_loss = meta_loss / self.grad_accumulation_steps
            meta_loss.backward()
            
            self.current_step += 1
            
            # 축적된 그래디언트로 업데이트
            if self.current_step % self.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                self.meta_scheduler.step()
            
            return meta_loss.item() * self.grad_accumulation_steps
            
        except Exception as e:
            print(f"Meta step 에러: {str(e)}")
            return None
    
    def get_current_lr(self):
        """현재 학습률 반환"""
        return self.meta_optimizer.param_groups[0]['lr']

class TimeSeriesDataset(torch.utils.data.Dataset):
    """시계열 데이터셋"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_preprocessed_data(file_path, sequence_length, is_normal=True):
    """전처리된 데이터 로드"""
    print(f"데이터 로드 중: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"원본 데이터 크기: {df.shape}")
    
    # LOAD 컬럼들만 사용
    load_columns = ['LOAD_1', 'LOAD_2', 'LOAD_3', 'LOAD_4', 'LOAD_5']
    sensor_columns = [col for col in load_columns if col in df.columns]
        
    sequences = []
    labels = []
    
    # sequence_id별로 시퀀스 생성
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].copy()
        seq_data = seq_data.sort_values('time_step')
        
        # 센서 데이터 추출
        sensor_data = seq_data[sensor_columns].values
        
        if len(sensor_data) == sequence_length:
            if not np.isnan(sensor_data).any() and not np.isinf(sensor_data).any():
                sequences.append(sensor_data.astype(np.float32))
                labels.append(0 if is_normal else 1)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"생성된 시퀀스: {sequences.shape}")
    print(f"레이블 분포: {np.bincount(labels)}")
    
    return sequences, labels

# 학습 곡선 시각화 함수
def plot_learning_curve(train_losses, val_losses, save_dir):
    """학습 곡선을 그리는 함수"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 로그 스케일 버전
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss (Log Scale)', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss (Log Scale)', linewidth=2)
    plt.title('Model Loss Over Epochs (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"학습 곡선이 {save_dir}/learning_curve.png에 저장되었습니다.")

# 성능 테이블 생성 함수
def create_performance_table(y_true, y_pred, save_dir):
    """성능 지표 테이블을 생성하는 함수"""
    # Confusion Matrix 계산
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 클래스별 지지도 계산
    support_normal = np.sum(y_true == 0)
    support_anomaly = np.sum(y_true == 1)
    
    # 테이블 데이터 생성
    table_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support (Normal)', 'Support (Anomaly)', 'True Positive', 'False Positive', 'True Negative', 'False Negative'],
        'Value': [f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', f'{support_normal}', f'{support_anomaly}', f'{tp}', f'{fp}', f'{tn}', f'{fn}']
    }
    
    # DataFrame 생성
    df = pd.DataFrame(table_data)
    
    # 테이블 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 테이블 생성
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 헤더 스타일링
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일링
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'performance_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV로도 저장
    df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)
    
    print(f"성능 테이블이 {save_dir}/performance_table.png에 저장되었습니다.")
    return df

# 히스토그램 생성 함수
def plot_score_distribution(y_true, anomaly_scores, threshold, save_dir):
    """이상 점수 분포 히스토그램을 그리는 함수"""
    # 정상/이상 데이터 분리
    normal_scores = anomaly_scores[y_true == 0]
    anomaly_scores_actual = anomaly_scores[y_true == 1]
    
    plt.figure(figsize=(15, 10))
    
    # 첫 번째 서브플롯: 전체 분포
    plt.subplot(2, 2, 1)
    plt.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='blue', density=True)
    plt.hist(anomaly_scores_actual, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores_actual)})', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distribution (All Data)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 두 번째 서브플롯: 로그 스케일
    plt.subplot(2, 2, 2)
    plt.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='blue', density=True)
    plt.hist(anomaly_scores_actual, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores_actual)})', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score (Log Scale)')
    plt.ylabel('Density')
    plt.title('Score Distribution (Log Scale)', fontweight='bold')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 세 번째 서브플롯: 정상 데이터만
    plt.subplot(2, 2, 3)
    plt.hist(normal_scores, bins=50, alpha=0.7, color='blue', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Normal Data Distribution (n={len(normal_scores)})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 네 번째 서브플롯: 이상 데이터만
    plt.subplot(2, 2, 4)
    plt.hist(anomaly_scores_actual, bins=50, alpha=0.7, color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Anomaly Data Distribution (n={len(anomaly_scores_actual)})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"점수 분포 히스토그램이 {save_dir}/score_distribution.png에 저장되었습니다.")

# Confusion Matrix 생성 함수
def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Confusion Matrix를 그리는 함수"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 정확도 정보 추가
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', 
             horizontalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion Matrix가 {save_dir}/confusion_matrix.png에 저장되었습니다.")
    
def evaluate_model(y_true, y_pred, anomaly_scores, threshold, save_dir):
    """모델 성능을 평가하고 모든 시각화를 생성하는 함수"""
    # 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 결과 출력
    print("\n=== 성능 평가 결과 ===")
    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"F1 점수: {f1:.4f}")
    print(f"임계값: {threshold:.6f}")
    
    # 분류 보고서
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])
    print("\n분류 보고서:")
    print(report)
    
    # 결과를 파일로 저장
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 성능 평가 결과 ===\n")
        f.write(f"정확도: {accuracy:.4f}\n")
        f.write(f"정밀도: {precision:.4f}\n")
        f.write(f"재현율: {recall:.4f}\n")
        f.write(f"F1 점수: {f1:.4f}\n")
        f.write(f"임계값: {threshold:.6f}\n\n")
        f.write("분류 보고서:\n")
        f.write(report)
    
    # 모든 시각화 생성
    print("\n=== 시각화 생성 중 ===")
    
    # 1. 성능 테이블 생성
    performance_df = create_performance_table(y_true, y_pred, save_dir)
    
    # 2. 점수 분포 히스토그램 생성
    plot_score_distribution(y_true, anomaly_scores, threshold, save_dir)
    
    # 3. Confusion Matrix 생성
    plot_confusion_matrix(y_true, y_pred, save_dir)
    
    print("=== 모든 시각화가 완료되었습니다 ===")
    
    return performance_df

def create_sequences_from_data(data, sequence_length, load_column):
    """특정 LOAD 컬럼으로 시퀀스 생성 - 5차원으로 수정"""
    sequences = []
        
    # LOAD_1~5 모든 컬럼 사용
    load_columns = ['LOAD_1', 'LOAD_2', 'LOAD_3', 'LOAD_4', 'LOAD_5']
    available_loads = [col for col in load_columns if col in data.columns]
    
    # sequence_id별로 처리
    for seq_id in data['sequence_id'].unique():
        seq_data = data[data['sequence_id'] == seq_id].copy()
        seq_data = seq_data.sort_values('time_steD_QTY')
        
        # 모든 LOAD 데이터 추출 (5차원)
        load_data = seq_data[available_loads].values  # (333, 5)
        
        if len(load_data) == sequence_length:
            if not np.isnan(load_data).any() and not np.isinf(load_data).any():
                sequences.append(load_data.astype(np.float32))
    
    return np.array(sequences) if sequences else np.array([])

def create_machine_load_meta_tasks(file_path, sequence_length=333, task_size=30, support_ratio=0.6):
    """기계별 + LOAD별 메타 태스크 생성 - 모든 LOAD 사용하되 태스크명만 구분"""
    tasks = []
    
    # 파일명에서 기계 ID 추출
    filename = os.path.basename(file_path)
    machine_id = filename.split('_')[0]
    
    print(f"처리 중인 기계: {machine_id}")
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # LOAD 컬럼들 확인
    load_columns = ['LOAD_1', 'LOAD_2', 'LOAD_3', 'LOAD_4', 'LOAD_5']
    available_loads = [col for col in load_columns if col in df.columns]
        
    machine_stats = []
    
    # 각 LOAD별로 태스크 생성 (데이터는 동일하지만 태스크 구분용)
    for load_col in available_loads:
        
        # 모든 LOAD 데이터로 시퀀스 생성 (5차원)
        sequences = create_sequences_from_data(df, sequence_length, load_col)
        
        # 태스크 생성
        task = create_single_task_simple(sequences, task_size, support_ratio)
        if task is not None:
            tasks.append(task)
            
            task_name = f"{machine_id}_{load_col}"
            
            machine_stats.append({
                'machine': machine_id,
                'load': load_col,
                'task_name': task_name,
                'sequences': len(sequences),
                'products': len(sequences)
            })
    
    return tasks, machine_stats

def create_all_machine_meta_tasks(data_dir, sequence_length=333, task_size=15, support_ratio=0.5):
    """모든 기계의 메타 태스크 생성"""
    import glob
    
    all_tasks = []
    all_stats = []
    
    # M???_normal_processed_softdtw.csv 파일들 찾기
    pattern_1 = os.path.join(data_dir, "M*_normal_processed_softdtw.csv")
    pattern_2 = os.path.join(data_dir, "M013_normal_processed_softdtw_targeted.csv")
    machine_files = glob.glob(pattern_1) + glob.glob(pattern_2)
    
    for file_path in machine_files:
        print(f"  {os.path.basename(file_path)}")
    
    # 각 기계별로 태스크 생성
    for file_path in machine_files:
        tasks, stats = create_machine_load_meta_tasks(
            file_path, sequence_length, task_size, support_ratio
        )
        
        all_tasks.extend(tasks)
        all_stats.extend(stats)
    
    return all_tasks, all_stats

def create_single_task_simple(sequences, task_size, support_ratio):
    """단일 태스크 생성"""
    if len(sequences) < task_size:
        return None
    
    # 랜덤하게 제품 선택
    indices = np.random.choice(len(sequences), task_size, replace=False)
    selected_sequences = sequences[indices]
    
    # Support/Query 분할
    support_size = int(task_size * support_ratio)
    support_data = selected_sequences[:support_size]
    query_data = selected_sequences[support_size:]
    
    if len(query_data) == 0:  # 안전장치
        query_data = support_data[-1:]
    
    # AutoEncoder용 더미 레이블
    support_labels = torch.zeros(len(support_data), dtype=torch.long)
    query_labels = torch.zeros(len(query_data), dtype=torch.long)
    
    return (
        torch.FloatTensor(support_data),
        support_labels,
        torch.FloatTensor(query_data),
        query_labels
    )

def load_test_data_with_split(file_path, sequence_length, first_split_ratio, second_split_ratio, is_normal=True):
    """테스트 데이터를 단계적으로 분할하여 로드"""
    print(f"테스트 데이터 단계적 분할 로드: {file_path}")
    
    # 전체 데이터 로드
    sequences, labels = load_preprocessed_data(file_path, sequence_length, is_normal)
    
    print(f"원본 데이터: {len(sequences)}개 시퀀스")
    
    # 1단계: 30% 분할
    if first_split_ratio < 1.0:
        sequences_subset, _, labels_subset, _ = train_test_split(
            sequences, labels, 
            test_size=1-first_split_ratio,
            random_state=42,
            stratify=labels if len(set(labels)) > 1 else None
        )
        print(f"1단계 분할 ({first_split_ratio*100}%): {len(sequences_subset)}개 시퀀스")
    else:
        sequences_subset, labels_subset = sequences, labels
    
    # 2단계: 20% 분할
    if second_split_ratio < 1.0:
        sequences_final, _, labels_final, _ = train_test_split(
            sequences_subset, labels_subset,
            test_size=1-second_split_ratio,
            random_state=42,
            stratify=labels_subset if len(set(labels_subset)) > 1 else None
        )
        print(f"2단계 분할 ({second_split_ratio*100}%): {len(sequences_final)}개 시퀀스")
    else:
        sequences_final, labels_final = sequences_subset, labels_subset
    
    final_ratio = first_split_ratio * second_split_ratio
    print(f"최종 사용률: {final_ratio*100:.1f}% ({len(sequences_final)}개 시퀀스)")
    
    return sequences_final, labels_final

def train_maml(model, meta_tasks, epochs, batch_size):
    """안전한 MAML 학습"""
    maml = MAML_LSTM_AE(model, meta_lr=0.005, inner_lr=0.01, inner_steps=3)
    
    meta_losses = []
    
    print(f"   총 태스크: {len(meta_tasks)}개")
    print(f"   각 에폭마다 모든 태스크 학습")
    print(f"   총 메타 업데이트: {epochs * len(meta_tasks)} 번")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        
        epoch_loss = 0.0
        success_count = 0
        error_count = 0
        
        for batch_idx in range(len(meta_tasks)):
            task_batch = [meta_tasks[batch_idx]]
            
            try:
                # 태스크 데이터 확인
                support_x, support_y, query_x, query_y = task_batch[0]
                
                # NaN/Inf 체크
                if torch.isnan(support_x).any() or torch.isnan(query_x).any():
                    error_count += 1
                    continue
                
                if torch.isinf(support_x).any() or torch.isinf(query_x).any():
                    error_count += 1
                    continue
                
                # 메타 학습 스텝
                batch_loss = maml.meta_train_step(task_batch)
                
                if batch_loss is None or np.isnan(batch_loss):
                    error_count += 1
                    continue
                
                epoch_loss += batch_loss
                success_count += 1
                
            except Exception as e:
                error_count += 1
                continue
        
        # 평균 손실 계산
        if success_count > 0:
            avg_loss = epoch_loss / success_count
            meta_losses.append(avg_loss)
            print(f"Meta Loss: {avg_loss:.6f}")
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    return maml, meta_losses

def fine_tune_maml(maml_model, finetune_data, finetune_labels, epochs, batch_size=8):
    """진짜 비지도학습 파인튜닝 - Normal만 사용"""
    
    # ✅ Normal 데이터만 사용 (Anomaly 완전 무시!)
    normal_mask = (finetune_labels == 0)
    normal_data = finetune_data[normal_mask]
    
    print(f"파인튜닝 시작")
    print(f"   Normal 데이터만 {len(normal_data)}개 사용")
    
    # 강화된 Normal 재구성 학습
    def enhanced_reconstruction_loss(reconstructed, original):
        """Normal 데이터를 완벽하게 재구성하도록 강화"""
        # MSE + L1 조합으로 더 정밀하게
        mse_loss = F.mse_loss(reconstructed, original)
        l1_loss = F.l1_loss(reconstructed, original)
        
        # 더 강한 압축을 위한 latent 정규화
        return mse_loss + 0.1 * l1_loss
    
    # Normal 데이터셋
    normal_dataset = TensorDataset(torch.FloatTensor(normal_data))
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
    
    # 적응적 학습률
    optimizer = optim.AdamW(maml_model.model.parameters(), lr=0.001, weight_decay=1e-5, amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    losses = []
    maml_model.model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_data, in normal_loader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            reconstructed = maml_model.model(batch_data)
            
            # ✅ Normal 데이터만으로 재구성 학습
            loss = enhanced_reconstruction_loss(reconstructed, batch_data)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(maml_model.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1:2d}/{epochs}] '
                  f'Normal Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}')
    
    print(f"✅ 순수 비지도학습 파인튜닝 완료!")    
    return losses
    
def detect_anomalies_maml(model, test_loader, threshold_percentile=98):
    """MAML 모델로 이상 탐지 - 수정된 버전"""
    model.eval()  # 평가 모드로 설정
    normal_scores = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            reconstructed = model(data)
            
            # 재구성 오차 계산
            mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2))
            
            # 정상 데이터의 점수만 별도 저장
            normal_mask = (labels == 0)
            normal_scores.extend(mse[normal_mask].cpu().numpy())
            
            # 전체 점수와 레이블 저장
            all_scores.extend(mse.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    normal_scores = np.array(normal_scores)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # ✅ 핵심 수정: Normal 데이터만으로 임계값 계산!
    threshold = np.percentile(normal_scores, threshold_percentile)
    predictions = (all_scores > threshold).astype(int)
    
    print(f"\n=== 임계값 계산 정보 ===")
    print(f"Normal 데이터 수: {len(normal_scores)}")
    print(f"Normal 점수 범위: {normal_scores.min():.6f} ~ {normal_scores.max():.6f}")
    print(f"Normal {threshold_percentile}퍼센타일 임계값: {threshold:.6f}")
    
    return all_scores, predictions, threshold

def main():
    parser = argparse.ArgumentParser(description='MAML LSTM AutoEncoder')
    parser.add_argument('--config', type=str, default='Hyper_parameter.json', help='설정 파일 경로')
    parser.add_argument('--train_data', type=str, default='Preprocessing_Data/M013_normal_processed_softdtw_targeted.csv', help='학습 데이터')
    parser.add_argument('--test_normal', type=str, default='Preprocessing_Data/M013_normal_processed_softdtw_targeted.csv', help='테스트 정상 데이터')
    parser.add_argument('--test_anomaly', type=str, default='Preprocessing_Data/M013_faulty_processed_softdtw_targeted.csv', help='테스트 이상 데이터')
    parser.add_argument('--meta_epochs', type=int, default=100, help='메타 학습 에폭')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='파인튜닝 에폭')
    
    args = parser.parse_args()
        
    # 하이퍼파라미터 로드
    config = load_hyperparameters(args.config)
    model_config = config['model_architecture']
    data_config = config['data_preprocessing']
    
    # MAML 전용 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    maml_root = "MULTI_MAML_LSTM_AE_Results_30%"
    os.makedirs(maml_root, exist_ok=True)
    
    current_run = f"run_{timestamp}"
    maml_run_dir = os.path.join(maml_root, current_run)
    os.makedirs(maml_run_dir, exist_ok=True)
    
    models_dir = os.path.join(maml_run_dir, "models")
    results_dir = os.path.join(maml_run_dir, "results")
    plots_dir = os.path.join(maml_run_dir, "plots")
    logs_dir = os.path.join(maml_run_dir, "logs")
    
    for dir_path in [models_dir, results_dir, plots_dir, logs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("="*80)
    print("MAML LSTM AutoEncoder 실행 - 30개 태스크 (6기계 × 5LOAD)")
    print("="*80)
    print(f"결과 저장 폴더: {maml_run_dir}")
    print("="*80)
    
    # 데이터 로드
    train_sequences, train_labels = load_preprocessed_data(
        args.train_data,
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    # 모델 생성 (input_dim=1, 각 LOAD별로 1차원)
    model = LSTM_AE(
        input_dim=5, 
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    # 초기 LSTM 가중치 최적화
    for module in model.modules():
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module.flatten_parameters()
        
    # 30개 메타 태스크 생성 (6기계 × 5LOAD)
    print("메타 태스크 생성 중...")
    
    data_dir = os.path.dirname(args.train_data)
    meta_tasks, task_stats = create_all_machine_meta_tasks(
        data_dir=data_dir,
        sequence_length=333,  # 제품 생산 과정 전체
        task_size=30,         # 태스크당 15개 제품
        support_ratio=0.6
    )
        
    # MAML 메타 학습
    maml_model, meta_losses = train_maml(
        model, 
        meta_tasks, 
        epochs=args.meta_epochs, 
        batch_size=2
    )
    
    # 파인튜닝용 데이터 준비 (실제 타겟 데이터의 일부)
    finetune_size = len(train_sequences)
    finetune_data = train_sequences[:finetune_size]
    finetune_labels = train_labels[:finetune_size]
    
    print(f"파인튜닝 데이터: {finetune_data.shape}")
    
    # MAML 모델 파인튜닝
    finetune_losses = fine_tune_maml(
        maml_model, 
        finetune_data, 
        finetune_labels, 
        epochs=args.finetune_epochs
    )
    
    # 모델 저장
    torch.save(maml_model.model.state_dict(), os.path.join(models_dir, 'maml_model.pth'))
    
    # 테스트 데이터 로드
    test_normal_sequences, test_normal_labels = load_test_data_with_split(
        args.test_normal,
        sequence_length=data_config['sequence_length'],
        first_split_ratio=0.3,
        second_split_ratio=0.2,
        is_normal=True
    )
    
    test_anomaly_sequences, test_anomaly_labels = load_preprocessed_data(
        args.test_anomaly,
        sequence_length=data_config['sequence_length'],
        is_normal=False
    )
    
    # 테스트 데이터 결합
    test_sequences = np.concatenate([test_normal_sequences, test_anomaly_sequences], axis=0)
    test_labels = np.concatenate([test_normal_labels, test_anomaly_labels], axis=0)
    
    # 테스트 데이터 로더
    test_dataset = TimeSeriesDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 이상 탐지 수행
    anomaly_scores, predictions, threshold = detect_anomalies_maml(
        maml_model.model, 
        test_loader
    )
    
    # 성능 평가
    performance_df = evaluate_model(
        y_true=test_labels,
        y_pred=predictions,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        save_dir=results_dir
    )
    
    # 결과 시각화에 파인튜닝 손실 추가
    plt.figure(figsize=(15, 10))
    
    # 메타 학습 손실
    plt.subplot(2, 3, 1)
    plt.plot(meta_losses)
    plt.title('MAML Meta Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Meta Loss')
    plt.grid(True)
    
    # 파인튜닝 손실
    plt.subplot(2, 3, 2)
    plt.plot(finetune_losses)
    plt.title('Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.grid(True)
    
    # 이상 점수 분포
    plt.subplot(2, 3, 3)
    normal_scores = anomaly_scores[test_labels == 0]
    anomaly_scores_plot = anomaly_scores[test_labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores_plot, bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # 성능 지표
    plt.subplot(2, 3, 4)
    metrics = performance_df['Metric'].tolist()
    values = performance_df['Value'].tolist()
    plt.bar(metrics, values)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'maml_lstm_ae_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 실행 요약 로그 저장
    summary_log = {
        'timestamp': timestamp,
        'config_used': config,
        'maml_parameters': {
            'meta_epochs': args.meta_epochs,
            'finetune_epochs': args.finetune_epochs,
            'num_meta_tasks': len(meta_tasks),
            'data_usage_ratio': 0.3,
            'finetune_data_size': len(finetune_data)
        },
        'training_losses': {
            'meta_losses': meta_losses,
            'finetune_losses': finetune_losses
        },
        'model_performance': performance_df.to_dict(),
        'final_threshold': float(threshold)
    }
    
    with open(os.path.join(logs_dir, 'maml_experiment_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_log, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("MAML LSTM AutoEncoder 완료!")
    print("="*80)
    print("학습 과정:")
    print(f"  1. 메타 학습: {len(meta_tasks)}개 태스크로 {args.meta_epochs} 에폭")
    print(f"  2. 파인튜닝: {len(finetune_data)}개 샘플로 {args.finetune_epochs} 에폭")
    
    # F1-Score 값을 안전하게 가져오기
    try:
        f1_row = performance_df[performance_df['Metric']=='F1-Score']
        if not f1_row.empty:
            f1_value = f1_row['Value'].iloc[0]
            # 문자열이면 float로 변환
            if isinstance(f1_value, str):
                f1_value = float(f1_value)
            print(f"  3. 최종 성능: F1-Score = {f1_value:.4f}")
        else:
            print(f"  3. 최종 성능: F1-Score를 찾을 수 없음")
    except Exception as e:
        print(f"  3. 최종 성능: F1-Score 출력 오류 - {e}")
    
    print("="*80)
    print(f"결과 폴더: {maml_run_dir}")
    print(f"  ├── models/maml_model.pth")
    print(f"  ├── results/performance_metrics.csv")
    print(f"  ├── plots/maml_lstm_ae_summary.png")
    print(f"  └── logs/maml_experiment_summary.json")
    print("="*80)

if __name__ == "__main__":
    main()