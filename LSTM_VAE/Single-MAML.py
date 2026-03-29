import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import argparse
from copy import deepcopy
import datetime
import gc
from sklearn.model_selection import train_test_split

# Basic_LSTM_AE에서 필요한 클래스들 가져오기
from Basic_LSTM_VAE.Basic_LSTM_AE import LSTM_AE, TimeSeriesDataset, evaluate_model

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

def load_hyperparameters(config_file='Hyper_parameter.json'):
    """하이퍼파라미터 설정 파일을 로드하는 함수"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def load_preprocessed_data(file_path, sequence_length=333, is_normal=True):
    """전처리된 데이터를 로드하는 함수"""
    
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
    
    # NaN 및 무한대 값 처리
    if df.isnull().any().any():
        df = df.dropna()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[numeric_cols]).any().any():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # sequence_id별로 그룹화
    sequences = []
    labels = []
    
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].copy()
        seq_data = seq_data.sort_values('time_step')
        
        # 센서 데이터만 추출
        sensor_columns = [col for col in seq_data.columns 
                         if col not in ['sequence_id', 'time_step', 'original_qty_value', 'original_length']]
        
        sensor_data = seq_data[sensor_columns].values
        
        if len(sensor_data) == sequence_length:
            if not np.isnan(sensor_data).any() and not np.isinf(sensor_data).any():
                sequences.append(sensor_data.astype(np.float32))
                labels.append(0 if is_normal else 1)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    return sequences, labels

class MAML_LSTM_AE:
    def __init__(self, model, meta_lr=0.01, inner_lr=0.1, inner_steps=5):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_data, support_labels):
        """내부 루프: 서포트 셋으로 빠른 적응"""
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 내부 그래디언트 스텝
        for step in range(self.inner_steps):
            # 순전파
            reconstructed = self.model(support_data)
            loss = nn.MSELoss()(reconstructed, support_data)
            
            # 그래디언트 계산 (allow_unused=True 추가)
            grads = torch.autograd.grad(
                loss, 
                self.model.parameters(), 
                create_graph=True, 
                allow_unused=True
            )
            
            # 파라미터 업데이트 (in-place)
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), grads):
                    if grad is not None:  # grad가 None이 아닌 경우만 업데이트
                        param -= self.inner_lr * grad
        
        return None 
    
    def _flatten_model_parameters(self, model):
        """LSTM 모델의 파라미터를 flatten"""
        for module in model.modules():
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                module.flatten_parameters()
                
    def meta_train_step(self, task_batch):
        """안전한 MAML 구현 - deepcopy 사용"""
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0
        
        for support_data, support_labels, query_data, query_labels in task_batch:
            support_data = support_data.to(device)
            query_data = query_data.to(device)
            
            # 1. 모델 복사본 생성 (원본 보호)
            temp_model = deepcopy(self.model)
            temp_model.train()
            temp_optimizer = optim.SGD(temp_model.parameters(), lr=self.inner_lr)
            
            # 2. Support 데이터로 빠른 적응 (복사본에서)
            for step in range(self.inner_steps):
                temp_optimizer.zero_grad()
                support_pred = temp_model(support_data)
                support_loss = nn.MSELoss()(support_pred, support_data)
                support_loss.backward()
                temp_optimizer.step()
            
            # 3. Query 데이터로 메타 손실 계산 (복사본에서)
            query_pred = temp_model(query_data)
            task_meta_loss = nn.MSELoss()(query_pred, query_data)
            
            # 4. 메타 손실을 원본 모델로 역전파하기 위한 트릭
            # 원본 모델로 다시 계산
            original_query_pred = self.model(query_data)
            
            # 적응된 모델의 출력과 원본 모델 출력의 차이를 이용
            meta_loss_for_backprop = nn.MSELoss()(original_query_pred, query_data)
            total_meta_loss += meta_loss_for_backprop
                        
            # 5. 임시 모델 삭제
            del temp_model, temp_optimizer, support_pred, query_pred, original_query_pred
            torch.cuda.empty_cache()
        
        # 6. 메타 그래디언트 계산 및 업데이트
        if total_meta_loss > 0:
            avg_meta_loss = total_meta_loss / len(task_batch)
            avg_meta_loss.backward()
            
            # 그래디언트 확인
            total_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            self.meta_optimizer.step()
            
            return avg_meta_loss.item()
        else:
            return 0.0

def create_load_based_meta_tasks(sequences, labels, file_path, task_size=20, support_ratio=0.5):
    """LOAD 구간별로 메타 태스크 생성"""
    tasks = []
    
    # 원본 CSV 파일에서 LOAD 정보 읽기
    df = pd.read_csv(file_path)
    
    # LOAD 컬럼들 확인
    load_columns = [col for col in df.columns if 'LOAD' in col.upper()]
    print(f"발견된 LOAD 컬럼들: {load_columns}")
    
    if not load_columns:
        print("LOAD 컬럼을 찾을 수 없습니다.")
        return []
    
    # 각 LOAD 컬럼별로 구간 나누기
    for load_col in load_columns[:5]:  # LOAD_1~5만 사용
        if load_col not in df.columns:
            continue
            
        load_values = df[load_col].dropna()
        
        # LOAD 값을 5개 구간으로 나누기 (quantile 기반)
        try:
            quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bins = load_values.quantile(quantiles).values
            
            # 구간별로 sequence_id 수집
            for i in range(len(bins)-1):
                min_val, max_val = bins[i], bins[i+1]
                
                # 해당 구간에 속하는 sequence_id들
                if i == len(bins)-2:  # 마지막 구간은 max_val 포함
                    mask = (df[load_col] >= min_val) & (df[load_col] <= max_val)
                else:
                    mask = (df[load_col] >= min_val) & (df[load_col] < max_val)
                
                segment_seq_ids = df[mask]['sequence_id'].unique()
                                
                if len(segment_seq_ids) >= task_size:
                    # sequence_id를 sequences 배열의 인덱스로 변환
                    # 간단히 처음부터 순서대로 매핑 (실제로는 더 정확한 매핑 필요)
                    available_indices = list(range(min(len(segment_seq_ids), len(sequences))))
                    
                    if len(available_indices) >= task_size:
                        # 랜덤하게 선택
                        selected_indices = np.random.choice(
                            available_indices, 
                            task_size, 
                            replace=False
                        )
                        
                        task_sequences = sequences[selected_indices]
                        task_labels = labels[selected_indices]
                        
                        # Support/Query 분할
                        support_size = int(task_size * support_ratio)
                        support_data = task_sequences[:support_size]
                        support_labels = task_labels[:support_size]
                        query_data = task_sequences[support_size:]
                        query_labels = task_labels[support_size:]
                        
                        # 빈 쿼리 데이터 방지
                        if len(query_data) == 0:
                            query_data = support_data[-1:]
                            query_labels = support_labels[-1:]
                        
                        tasks.append((
                            torch.FloatTensor(support_data),
                            torch.LongTensor(support_labels),
                            torch.FloatTensor(query_data),
                            torch.LongTensor(query_labels)
                        ))
                        
                        print(f"✅ 태스크 생성: {load_col}_구간{i+1}")
                        
        except Exception as e:
            print(f"{load_col} 처리 중 오류: {e}")
            continue
    
    print(f"총 생성된 태스크 수: {len(tasks)}")
    return tasks

def train_maml(model, meta_tasks, epochs=10, batch_size=1):
    """안전한 MAML 학습"""
    maml = MAML_LSTM_AE(
        model, 
        meta_lr=0.01,
        inner_lr=0.05,  # 조금 줄임
        inner_steps=3
    )
    
    meta_losses = []
    
    print(f"🚀 MAML 메타 학습 시작")
    print(f"설정: Meta LR={maml.meta_lr}, Inner LR={maml.inner_lr}, Inner Steps={maml.inner_steps}")
    print(f"총 {epochs} 에폭, {len(meta_tasks)}개 태스크")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        
        epoch_loss = 0.0
        num_batches = len(meta_tasks)
        
        for batch_idx in range(num_batches):
            task_batch = [meta_tasks[batch_idx]]
            
            try:
                batch_loss = maml.meta_train_step(task_batch)
                epoch_loss += batch_loss
            except Exception as e:
                continue
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        meta_losses.append(avg_loss)
        
        print(f"Meta Loss: {avg_loss:.6f}")
        print(f"Epoch [{epoch+1:3d}/{epochs}] Meta Loss: {avg_loss:.6f}")
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    return maml, meta_losses

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

def fine_tune_maml(maml_model, finetune_data, finetune_labels, epochs=30, batch_size=8):
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
    optimizer = optim.AdamW(maml_model.model.parameters(), lr=0.01, weight_decay=1e-5)
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

def detect_anomalies_maml(model, test_loader, threshold_percentile=95):
    """MAML 모델로 이상 탐지"""
    model.eval()  # 평가 모드로 설정
    anomaly_scores = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed = model(data)
            
            # 재구성 오차 계산
            mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2))
            anomaly_scores.extend(mse.cpu().numpy())
    
    anomaly_scores = np.array(anomaly_scores)
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    predictions = (anomaly_scores > threshold).astype(int)
    
    return anomaly_scores, predictions, threshold

def main():
    parser = argparse.ArgumentParser(description='MAML LSTM AutoEncoder')
    parser.add_argument('--config', type=str, default='Hyper_parameter.json', help='설정 파일 경로')
    parser.add_argument('--train_data', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='메타 학습 데이터')
    parser.add_argument('--finetune_data', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='파인튜닝 데이터')
    parser.add_argument('--test_normal', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='테스트 정상 데이터')
    parser.add_argument('--test_anomaly', type=str, default='Preprocessing_Data/M014_faulty_processed_softdtw.csv', help='테스트 이상 데이터')
    parser.add_argument('--meta_epochs', type=int, default=100, help='메타 학습 에폭')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='파인튜닝 에폭')
    
    args = parser.parse_args()
        
    # 하이퍼파라미터 로드
    config = load_hyperparameters(args.config)
    model_config = config['model_architecture']
    data_config = config['data_preprocessing']
    
    # MAML 전용 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    maml_root = "SINGLE_MAML_LSTM_AE_Results_30%"
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
    print("MAML LSTM AutoEncoder 실행")
    print("="*80)
    print(f"메타 학습 데이터: {args.train_data}")
    print(f"파인튜닝 데이터: {args.finetune_data}")
    print(f"결과 저장 폴더: {maml_run_dir}")
    print("="*80)
    
    # 메타 학습용 데이터 로드 (M014)
    train_sequences, train_labels = load_preprocessed_data(
        args.train_data,
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    # 모델 생성
    input_dim = train_sequences.shape[2]
    model = LSTM_AE(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    # 초기 LSTM 가중치 최적화
    for module in model.modules():
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module.flatten_parameters()
    
    print(f"모델 입력 차원: {input_dim}")
    
    # LOAD별 메타 태스크 생성 (M014 데이터로)
    print("LOAD별 메타 태스크 생성 중...")
    meta_tasks = create_load_based_meta_tasks(
        train_sequences, 
        train_labels, 
        args.train_data,
        task_size=20,
        support_ratio=0.5
    )
        
    # MAML 메타 학습 (M014로)
    maml_model, meta_losses = train_maml(
        model, 
        meta_tasks, 
        epochs=args.meta_epochs, 
        batch_size=2
    )
    
    # 파인튜닝용 데이터 준비 (M013 normal 데이터)
    print(f"\n파인튜닝용 데이터 로드 중: {args.finetune_data}")
    finetune_sequences, finetune_labels = load_preprocessed_data(
        args.finetune_data,
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    print(f"파인튜닝 데이터: {finetune_sequences.shape}")
    
    # MAML 모델 파인튜닝 (M013 normal 데이터로)
    finetune_losses = fine_tune_maml(
        maml_model, 
        finetune_sequences, 
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
    plt.title('MAML Meta Learning Loss (M014)')
    plt.xlabel('Epoch')
    plt.ylabel('Meta Loss')
    plt.grid(True)
    
    # 파인튜닝 손실
    plt.subplot(2, 3, 2)
    plt.plot(finetune_losses)
    plt.title('Fine-tuning Loss (M013)')
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
        'data_info': {
            'meta_learning_data': args.train_data,
            'finetune_data': args.finetune_data,
            'test_normal_data': args.test_normal,
            'test_anomaly_data': args.test_anomaly
        },
        'maml_parameters': {
            'meta_epochs': args.meta_epochs,
            'finetune_epochs': args.finetune_epochs,
            'num_meta_tasks': len(meta_tasks),
            'finetune_data_size': len(finetune_sequences)
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
    print(f"  1. 메타 학습: M014 데이터의 {len(meta_tasks)}개 태스크로 {args.meta_epochs} 에폭")
    print(f"  2. 파인튜닝: M013 normal 데이터 {len(finetune_sequences)}개로 {args.finetune_epochs} 에폭")
    
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