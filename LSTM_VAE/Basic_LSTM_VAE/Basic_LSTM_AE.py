import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
import pickle
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# GPU 사용 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")

print(f"사용 장치: {device}")

# 커스텀 데이터셋 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]]).squeeze()
        else:
            return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([0.0]).squeeze()

# LSTM-AE 모델 정의
class LSTM_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout_rate=0.2):
        super(LSTM_AE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # 인코더
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 인코더에서 잠재 공간으로
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # 잠재 공간에서 디코더로
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        
        # 디코더
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    
    def encode(self, x):
        # 인코더 LSTM 통과
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-1]  # 마지막 레이어의 hidden state
        
        # 잠재 공간으로 매핑
        latent = self.encoder_fc(hidden)
        
        return latent
    
    def decode(self, latent, seq_len):
        # 잠재 공간에서 hidden 차원으로 변환
        hidden = self.decoder_fc(latent)
        
        # 시퀀스 길이만큼 반복
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 디코더 LSTM 통과
        outputs, _ = self.decoder_lstm(hidden)
        
        # 출력 레이어 통과
        reconstructed = self.output_layer(outputs)
        
        return reconstructed
    
    def forward(self, x):
        # 인코딩
        latent = self.encode(x)
        
        # 디코딩
        reconstructed = self.decode(latent, x.size(1))
        
        return reconstructed
    
    def compute_loss(self, x, reconstructed):
        # 재구성 손실 (MSE) - 배치 평균으로 정규화
        recon_loss = torch.mean((reconstructed - x) ** 2)
        
        return recon_loss

def load_hyperparameters(json_path="Hyper_parameter.json"):
    """JSON 파일에서 하이퍼파라미터를 로드하는 함수"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        print(f"하이퍼파라미터를 {json_path}에서 로드했습니다.")
        return params
    except FileNotFoundError:
        print(f"{json_path} 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
        return None

def save_hyperparameters(args, save_dir, additional_params=None):
    """사용된 하이퍼파라미터를 JSON 파일로 저장하는 함수"""
    params = {
        "model_architecture": {
            "model_type": "LSTM_AE",
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "dropout_rate": args.dropout_rate
        },
        "training_parameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "sequence_length": args.sequence_length,
            "production_cycle": args.production_cycle
        },
        "data_paths": {
            "train_data": args.train_data,
            "test_normal_data": args.test_normal_data,
            "test_anomaly_data": args.test_anomaly_data
        },
        "directories": {
            "save_dir": args.save_dir,
            "results_dir": args.results_dir
        }
    }
    
    # 추가 파라미터가 있으면 병합
    if additional_params:
        params.update(additional_params)
    
    # JSON 파일로 저장
    json_path = os.path.join(save_dir, 'used_hyperparameters.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"사용된 하이퍼파라미터가 {json_path}에 저장되었습니다.")

def load_preprocessed_data(file_path, sequence_length=333, is_normal=True):
    """전처리된 데이터를 로드하는 함수"""
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 제거할 컬럼들 정의
    columns_to_remove = [
        'VIBRATION_VECTOR', 'SPINDLE_TEMP', 'SERVOTEMP_1', 'SERVOTEMP_2', 
        'SERVOTEMP_3', 'SERVOTEMP_4', 'SERVOTEMP_5', 'ALARM_CODE'
    ]
    
    # 제거할 컬럼들이 있으면 제거
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # NaN 체크
    if df.isnull().any().any():
        df = df.dropna()
    
    # 무한대 값 체크
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[numeric_cols]).any().any():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # sequence_id별로 그룹화
    sequences = []
    labels = []
    
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].copy()
        seq_data = seq_data.sort_values('time_step')
        
        # 메타데이터 컬럼 제외하고 센서 데이터만 추출
        sensor_columns = [col for col in seq_data.columns 
                         if col not in ['sequence_id', 'time_step', 'original_qty_value', 'original_length']]
        
        sensor_data = seq_data[sensor_columns].values
        
        # 시퀀스 길이 확인
        if len(sensor_data) == sequence_length:
            # 데이터 검증
            if not np.isnan(sensor_data).any() and not np.isinf(sensor_data).any():
                sequences.append(sensor_data.astype(np.float32))
                labels.append(0 if is_normal else 1)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
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

# 학습 함수
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, save_dir='saved_models'):
    """LSTM-AE 모델을 학습하는 함수 (성능 최적화)"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Mixed precision 사용 (GPU 메모리 효율성 향상)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"에폭 {epoch+1}/{epochs} 학습")):
            data = data.to(device, non_blocking=True)  # non_blocking으로 전송 최적화
            
            optimizer.zero_grad()
            
            # Mixed precision 사용
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    reconstructed = model(data)
                    loss = model.compute_loss(data, reconstructed)
                
                # 손실이 비정상적이면 건너뛰기 (더 효율적인 검사)
                if not torch.isfinite(loss):
                    continue
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed = model(data)
                loss = model.compute_loss(data, reconstructed)
                
                if not torch.isfinite(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        
        # 검증
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        reconstructed = model(data)
                        loss = model.compute_loss(data, reconstructed)
                else:
                    reconstructed = model(data)
                    loss = model.compute_loss(data, reconstructed)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"에폭 {epoch+1}: 학습 손실: {avg_train_loss:.6f}, 검증 손실: {avg_val_loss:.6f}")
        
        # 학습률 스케줄러
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"학습률이 {old_lr:.6f}에서 {new_lr:.6f}로 감소했습니다.")
        
        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 모델 저장
            model_save_path = os.path.join(save_dir, 'best_lstm_ae_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'latent_dim': model.latent_dim,
                    'num_layers': model.num_layers,
                    'dropout_rate': 0.2
                },
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, model_save_path)
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= early_stopping_patience:
            print(f"조기 종료: {early_stopping_patience} 에폭 동안 개선되지 않음")
            break
    
    # 학습 완료 후 학습 곡선 그리기
    plot_learning_curve(train_losses, val_losses, save_dir)
    
    return model, train_losses, val_losses

# 이상 탐지 함수
def detect_anomalies(model, test_loader, threshold=None):
    """이상을 탐지하는 함수"""
    model.eval()
    reconstruction_errors = []
    true_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(test_loader, desc="이상 탐지 중")):
            data = data.to(device)
            
            # 모델 통과
            reconstructed = model(data)
            
            # 재구성 오차 계산 (MSE)
            mse_loss = torch.mean((reconstructed - data) ** 2, dim=(1, 2))
            
            # CPU로 이동하고 numpy 배열로 변환
            reconstruction_errors.extend(mse_loss.cpu().numpy())
            true_labels.extend(labels.view(-1).cpu().numpy())
    
    # numpy 배열로 변환
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)
    
    # 임계값 계산
    if threshold is None:
        normal_mask = (true_labels == 0)
        if np.any(normal_mask):
            normal_errors = reconstruction_errors[normal_mask]
            # 평균 + 3*표준편차를 임계값으로 설정
            threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
        else:
            threshold = np.percentile(reconstruction_errors, 95)
    
    # 이상 탐지
    predictions = (reconstruction_errors > threshold).astype(int)
    
    return reconstruction_errors, predictions, threshold

def load_test_data_with_split(file_path, sequence_length, first_split_ratio=0.3, second_split_ratio=0.2, is_normal=True):
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

# 모델 평가 함수
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

def main():
    parser = argparse.ArgumentParser(description='Basic LSTM AutoEncoder with Preprocessed Data')
    parser.add_argument('--config', type=str, default='Hyper_parameter.json', help='설정 파일 경로')
    parser.add_argument('--train_data', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='학습 데이터')
    parser.add_argument('--test_normal', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='테스트 정상 데이터')
    parser.add_argument('--test_anomaly', type=str, default='Preprocessing_Data/M014_faulty_processed_softdtw.csv', help='테스트 이상 데이터')
    parser.add_argument('--data_ratio', type=float, default=0.3, help='M014 파인튜닝 데이터 사용 비율 (기본값: 0.3)')
    
    args = parser.parse_args()
    
    # 하이퍼파라미터 로드
    config = load_hyperparameters(args.config)
    model_config = config['model_architecture']
    train_config = config['training_parameters']
    data_config = config['data_preprocessing']
    
    # Basic LSTM AE 전용 디렉토리 생성
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Basic LSTM AE 루트 폴더
    basic_root = "Basic_LSTM_AE_Results_30%"
    os.makedirs(basic_root, exist_ok=True)
    
    # 각 실행별 폴더 (타임스탬프 포함)
    current_run = f"run_{timestamp}"
    basic_run_dir = os.path.join(basic_root, current_run)
    os.makedirs(basic_run_dir, exist_ok=True)
    
    # 세부 폴더들
    models_dir = os.path.join(basic_run_dir, "models")
    results_dir = os.path.join(basic_run_dir, "results")
    plots_dir = os.path.join(basic_run_dir, "plots")
    logs_dir = os.path.join(basic_run_dir, "logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("="*80)
    print("Basic LSTM AutoEncoder 실행")
    print("="*80)
    print(f"결과 저장 폴더: {basic_run_dir}")
    print("="*80)
    
    # 데이터 로드
    train_sequences, train_labels = load_preprocessed_data(
        args.train_data,
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    if args.data_ratio < 1.0:
        total_sequences = len(train_sequences)
        reduced_size = int(total_sequences * args.data_ratio)
        
        # 랜덤하게 선택하되 재현 가능하도록 시드 설정
        np.random.seed(42)
        selected_indices = np.random.choice(total_sequences, reduced_size, replace=False)
        selected_indices = np.sort(selected_indices)  # 순서 유지
        
        train_sequences = train_sequences[selected_indices]
        train_labels = train_labels[selected_indices]
        
        print(f"M014 학습 데이터 축소: {total_sequences} → {len(train_sequences)} ({args.data_ratio*100:.1f}%)")
        
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
    
    # 학습/검증 데이터 분할
    train_size = int(len(train_sequences) * train_config['train_ratio'])
    train_seq = train_sequences[:train_size]
    train_lab = train_labels[:train_size]
    val_seq = train_sequences[train_size:]
    val_lab = train_labels[train_size:]
    
    print(f"학습 데이터: {train_seq.shape}")
    print(f"검증 데이터: {val_seq.shape}")
    print(f"테스트 정상: {test_normal_sequences.shape}")
    print(f"테스트 이상: {test_anomaly_sequences.shape}")
    
    # 데이터 로더 생성
    train_dataset = TimeSeriesDataset(train_seq, train_lab)
    val_dataset = TimeSeriesDataset(val_seq, val_lab)
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'])
    
    # 모델 생성
    input_dim = train_seq.shape[2]
    model = LSTM_AE(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    print(f"모델 입력 차원: {input_dim}")
    
    # 모델 학습
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config['epochs'],
        learning_rate=train_config['learning_rate'],
        save_dir=models_dir
    )
    
    # 테스트 데이터 결합
    test_sequences = np.concatenate([test_normal_sequences, test_anomaly_sequences], axis=0)
    test_labels = np.concatenate([test_normal_labels, test_anomaly_labels], axis=0)
    
    # 테스트 데이터 로더
    test_dataset = TimeSeriesDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'])
    
    # 이상 탐지 수행
    anomaly_scores, predictions, threshold = detect_anomalies(
        model=model,
        test_loader=test_loader
    )
    
    # 성능 평가
    performance_df = evaluate_model(
        y_true=test_labels,
        y_pred=predictions,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        save_dir=results_dir
    )
    
    # 결과 시각화 및 저장
    plt.figure(figsize=(15, 10))
    
    # 학습 손실
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 이상 점수 분포
    plt.subplot(2, 3, 2)
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
    plt.subplot(2, 3, 3)
    metrics = performance_df['Metric'].tolist()
    values = performance_df['Value'].tolist()
    plt.bar(metrics, values)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 혼동 행렬
    plt.subplot(2, 3, 4)
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 시간별 이상 점수
    plt.subplot(2, 3, 5)
    plt.plot(anomaly_scores, alpha=0.7)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Anomaly Scores Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'basic_lstm_ae_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 실행 요약 로그 저장
    summary_log = {
        'timestamp': timestamp,
        'config_used': config,
        'data_files': {
            'train_data': args.train_data,
            'test_normal': args.test_normal,
            'test_anomaly': args.test_anomaly
        },
        'model_performance': performance_df.to_dict(),
        'final_threshold': float(threshold),
        'data_statistics': {
            'train_sequences': len(train_seq),
            'val_sequences': len(val_seq),
            'test_normal_sequences': len(test_normal_sequences),
            'test_anomaly_sequences': len(test_anomaly_sequences)
        }
    }
    
    with open(os.path.join(logs_dir, 'experiment_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_log, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Basic LSTM AutoEncoder 완료!")
    print("="*80)
    print(f"결과 폴더: {basic_run_dir}")
    print(f"  ├── models/best_model.pth")
    print(f"  ├── results/performance_metrics.csv")
    print(f"  ├── plots/basic_lstm_ae_summary.png")
    print(f"  └── logs/experiment_summary.json")
    print("="*80)

if __name__ == "__main__":
    main()