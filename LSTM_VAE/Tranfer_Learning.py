import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import argparse
import json
from sklearn.model_selection import train_test_split

# Basic_LSTM_AE.py에서 필요한 클래스들 가져오기
from Basic_LSTM_VAE.Basic_LSTM_AE import LSTM_AE, TimeSeriesDataset, train_model, detect_anomalies, evaluate_model

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
    removed_cols = []
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_cols.append(col)
    
    
    # sequence_id 컬럼 확인
    if 'sequence_id' not in df.columns:
        # 전체 데이터를 하나의 시퀀스로 처리
        sensor_columns = [col for col in df.columns 
                         if col not in ['time_step', 'original_qty_value', 'original_length']]
        
        sensor_data = df[sensor_columns].values
        
        if len(sensor_data) >= sequence_length:
            # 시퀀스 길이만큼 자르기
            sensor_data = sensor_data[:sequence_length]
            sequences = np.array([sensor_data], dtype=np.float32)
            labels = np.array([0 if is_normal else 1], dtype=np.int64)
            
            return sequences, labels
        else:
            return np.array([]), np.array([])
    
    # NaN 체크
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        df = df.dropna()
    
    # 무한대 값 체크
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[numeric_cols]).any().any():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # sequence_id별로 그룹화
    sequences = []
    labels = []
    
    unique_seq_ids = df['sequence_id'].unique()
    
    valid_sequences = 0
    invalid_sequences = 0
    
    for seq_id in unique_seq_ids:
        seq_data = df[df['sequence_id'] == seq_id].copy()
        seq_data = seq_data.sort_values('time_step')
                
        # 메타데이터 컬럼 제외하고 센서 데이터만 추출
        sensor_columns = [col for col in seq_data.columns 
                         if col not in ['sequence_id', 'time_step', 'original_qty_value', 'original_length']]
        
        if len(sensor_columns) == 0:
            invalid_sequences += 1
            continue
        
        sensor_data = seq_data[sensor_columns].values
        
        # 시퀀스 길이 확인
        if len(sensor_data) == sequence_length:
            # 데이터 검증
            if not np.isnan(sensor_data).any() and not np.isinf(sensor_data).any():
                # 데이터 정규화 (혹시 안되어 있을 경우)
                if np.abs(sensor_data).max() > 10:  # 정규화가 안된 것 같으면
                    print(f"    데이터 정규화 수행")
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    sensor_data = scaler.fit_transform(sensor_data)
                
                sequences.append(sensor_data.astype(np.float32))
                labels.append(0 if is_normal else 1)
                valid_sequences += 1
            else:
                invalid_sequences += 1
        else:
            invalid_sequences += 1
    
    print(f"유효한 시퀀스: {valid_sequences}, 무효한 시퀀스: {invalid_sequences}")
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    return sequences, labels

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

def transfer_learning_finetune(pretrained_model, finetune_data_loader, val_data_loader, 
                              epochs=20, learning_rate=0.00001, save_dir='transfer_models'):
    """사전 학습된 모델을 파인튜닝하는 함수"""
    
    os.makedirs(save_dir, exist_ok=True)

    
    print(f"\n파인튜닝 시작 (에폭: {epochs}, 학습률: {learning_rate})")
    
    def enhanced_reconstruction_loss(reconstructed, original):
        """Normal 데이터를 완벽하게 재구성하도록 강화"""
        # MSE + L1 조합으로 더 정밀하게
        mse_loss = F.mse_loss(reconstructed, original)
        l1_loss = F.l1_loss(reconstructed, original)
        
        # 더 강한 압축을 위한 latent 정규화
        return mse_loss + 0.1 * l1_loss
    
    # 옵티마이저 (낮은 학습률 사용)
    optimizer = optim.AdamW(pretrained_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 학습 모드
        pretrained_model.train()
        train_loss = 0.0
        
        for data, _ in finetune_data_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed = pretrained_model(data)
            loss = enhanced_reconstruction_loss(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(finetune_data_loader)
        train_losses.append(avg_train_loss)
        
        # 검증
        pretrained_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_data_loader:
                data = data.to(device)
                reconstructed = pretrained_model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_data_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # 모델 설정 정보 수집 (안전하게)
            model_config = {
                'input_dim': getattr(pretrained_model, 'input_dim', None),
                'hidden_dim': getattr(pretrained_model, 'hidden_dim', None),
                'latent_dim': getattr(pretrained_model, 'latent_dim', None),
                'num_layers': getattr(pretrained_model, 'num_layers', None)
            }
            
            # dropout_rate는 있을 경우에만 추가
            if hasattr(pretrained_model, 'dropout_rate'):
                model_config['dropout_rate'] = pretrained_model.dropout_rate
            elif hasattr(pretrained_model, 'dropout'):
                model_config['dropout_rate'] = pretrained_model.dropout
            else:
                model_config['dropout_rate'] = 0.0  # 기본값
            
            torch.save({
                'model_state_dict': pretrained_model.state_dict(),
                'model_config': model_config
            }, os.path.join(save_dir, 'transfer_model.pth'))
    
    return pretrained_model, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning with LSTM AutoEncoder')
    parser.add_argument('--config', type=str, default='Hyper_parameter.json', help='설정 파일 경로')
    parser.add_argument('--m001_data', type=str, default='Preprocessing_Data/M001_normal_processed_softdtw.csv', help='M001 사전 학습 데이터')
    parser.add_argument('--m014_train_data', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='M014 파인튜닝 데이터')
    parser.add_argument('--m014_test_normal', type=str, default='Preprocessing_Data/M014_normal_processed_softdtw.csv', help='M014 테스트 정상 데이터')
    parser.add_argument('--m014_test_anomaly', type=str, default='Preprocessing_Data/M014_faulty_processed_softdtw.csv', help='M014 테스트 이상 데이터')
    parser.add_argument('--m014_data_ratio', type=float, default=0.3, help='M014 파인튜닝 데이터 사용 비율 (기본값: 0.3)')
    
    args = parser.parse_args()
    
    # 하이퍼파라미터 로드
    config = load_hyperparameters(args.config)
    model_config = config['model_architecture']
    train_config = config['training_parameters']
    data_config = config['data_preprocessing']
    paths_config = config['paths']
    
    # Transfer Learning 전용 디렉토리 생성
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Transfer Learning 루트 폴더
    transfer_root = "Transfer_Learning_Results_30%"
    os.makedirs(transfer_root, exist_ok=True)
    
    # 각 실행별 폴더 (타임스탬프 포함)
    current_run = f"run_{timestamp}"
    transfer_run_dir = os.path.join(transfer_root, current_run)
    os.makedirs(transfer_run_dir, exist_ok=True)
    
    # 세부 폴더들
    models_dir = os.path.join(transfer_run_dir, "models")
    results_dir = os.path.join(transfer_run_dir, "results")
    plots_dir = os.path.join(transfer_run_dir, "plots")
    logs_dir = os.path.join(transfer_run_dir, "logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("="*80)
    print("Transfer Learning 폴더 구조")
    print("="*80)
    print(f"루트 폴더: {transfer_root}")
    print(f"현재 실행: {transfer_run_dir}")
    print(f"  ├── models/     : 학습된 모델 저장")
    print(f"  ├── results/    : 성능 결과 및 CSV")
    print(f"  ├── plots/      : 그래프 및 시각화")
    print(f"  └── logs/       : 학습 로그")
    print("="*80)
    
    # 출력 디렉토리 생성
    os.makedirs(paths_config['save_dir'], exist_ok=True)
    os.makedirs(paths_config['results_dir'], exist_ok=True)
    
    print("="*60)
    print("1단계: M001 데이터로 사전 학습")
    print("="*60)
    
    # M001 데이터 로드
    m001_sequences, m001_labels = load_preprocessed_data(
        args.m001_data, 
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    # 학습/검증 분할
    train_size = int(len(m001_sequences) * train_config['train_ratio'])
    train_sequences = m001_sequences[:train_size]
    train_labels = m001_labels[:train_size]
    val_sequences = m001_sequences[train_size:]
    val_labels = m001_labels[train_size:]
    
    print(f"M001 학습 데이터: {train_sequences.shape}")
    print(f"M001 검증 데이터: {val_sequences.shape}")
    
    # 데이터 로더 생성
    train_dataset = TimeSeriesDataset(train_sequences, train_labels)
    val_dataset = TimeSeriesDataset(val_sequences, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'])
    
    # M001 모델 생성 및 학습
    input_dim = train_sequences.shape[2]
    model = LSTM_AE(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    # M001 사전 학습
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config['epochs'],
        learning_rate=train_config['learning_rate'],
        save_dir=models_dir  # Transfer Learning 모델 폴더
    )
    
    print("M001 사전 학습 완료!")
    
    print("\n" + "="*60)
    print("2단계: M014 데이터로 파인튜닝")
    print("="*60)
    
    # M014 파인튜닝 데이터 로드
    m014_sequences, m014_labels = load_preprocessed_data(
        args.m014_train_data,
        sequence_length=data_config['sequence_length'],
        is_normal=True
    )
    
    # M014 파인튜닝 데이터를 지정된 비율로 줄이기
    if args.m014_data_ratio < 1.0:
        total_m014_sequences = len(m014_sequences)
        reduced_m014_size = int(total_m014_sequences * args.m014_data_ratio)
        
        # 랜덤하게 선택하되 재현 가능하도록 시드 설정
        np.random.seed(42)
        selected_m014_indices = np.random.choice(total_m014_sequences, reduced_m014_size, replace=False)
        selected_m014_indices = np.sort(selected_m014_indices)  # 순서 유지
        
        m014_sequences = m014_sequences[selected_m014_indices]
        m014_labels = m014_labels[selected_m014_indices]
        
        print(f"M014 파인튜닝 데이터 축소: {total_m014_sequences} → {len(m014_sequences)} ({args.m014_data_ratio*100:.1f}%)")
        
    # 파인튜닝 데이터 분할
    finetune_train_size = int(len(m014_sequences) * train_config['train_ratio'])
    finetune_train_sequences = m014_sequences[:finetune_train_size]
    finetune_train_labels = m014_labels[:finetune_train_size]
    finetune_val_sequences = m014_sequences[finetune_train_size:]
    finetune_val_labels = m014_labels[finetune_train_size:]
    
    print(f"M014 파인튜닝 학습 데이터: {finetune_train_sequences.shape}")
    print(f"M014 파인튜닝 검증 데이터: {finetune_val_sequences.shape}")
    
    # 파인튜닝 데이터 로더
    finetune_train_dataset = TimeSeriesDataset(finetune_train_sequences, finetune_train_labels)
    finetune_val_dataset = TimeSeriesDataset(finetune_val_sequences, finetune_val_labels)
    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=train_config['batch_size'])
    
    # 파인튜닝 수행 (낮은 학습률과 적은 에폭)
    finetune_epochs = max(10, train_config['epochs'] // 5)  # 원래 에폭의 1/5
    finetune_lr = train_config['learning_rate'] / 10  # 원래 학습률의 1/10
    
    transfer_model, finetune_train_losses, finetune_val_losses = transfer_learning_finetune(
        pretrained_model=model,
        finetune_data_loader=finetune_train_loader,
        val_data_loader=finetune_val_loader,
        epochs=finetune_epochs,
        learning_rate=finetune_lr,
        save_dir=models_dir  # Transfer Learning 모델 폴더
    )
    
    print("M014 파인튜닝 완료!")
    
    print("\n" + "="*60)
    print("3단계: M014 테스트 및 이상 탐지")
    print("="*60)
    
    # M014 테스트 데이터 로드
    test_normal_sequences, test_normal_labels = load_test_data_with_split(
        args.m014_test_normal,
        sequence_length=data_config['sequence_length'],
        first_split_ratio=0.3,
        second_split_ratio=0.2,
        is_normal=True
    )
    
    test_anomaly_sequences, test_anomaly_labels = load_preprocessed_data(
        args.m014_test_anomaly,
        sequence_length=data_config['sequence_length'],
        is_normal=False
    )
    
    # 테스트 데이터 결합
    test_sequences = np.concatenate([test_normal_sequences, test_anomaly_sequences], axis=0)
    test_labels = np.concatenate([test_normal_labels, test_anomaly_labels], axis=0)
    
    print(f"M014 테스트 데이터: {test_sequences.shape}")
    print(f"정상 샘플: {np.sum(test_labels == 0)}")
    print(f"이상 샘플: {np.sum(test_labels == 1)}")
    
    # 테스트 데이터 로더
    test_dataset = TimeSeriesDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'])
    
    # 이상 탐지 수행
    anomaly_scores, predictions, threshold = detect_anomalies(
        model=transfer_model,
        test_loader=test_loader
    )
    
    # 성능 평가
    performance_df = evaluate_model(
        y_true=test_labels,
        y_pred=predictions,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        save_dir=results_dir  # Transfer Learning 결과 폴더
    )
    
    # 학습 손실 그래프 저장
    plt.figure(figsize=(12, 8))
    
    # 사전 학습 손실
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='M001 Train Loss')
    plt.plot(val_losses, label='M001 Val Loss')
    plt.title('M001 Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 파인튜닝 손실
    plt.subplot(2, 2, 2)
    plt.plot(finetune_train_losses, label='M014 Train Loss')
    plt.plot(finetune_val_losses, label='M014 Val Loss')
    plt.title('M014 Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 이상 점수 분포
    plt.subplot(2, 2, 3)
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
    
    # 성능 지표 바 차트
    plt.subplot(2, 2, 4)
    metrics = performance_df['Metric'].tolist()
    values = performance_df['Value'].tolist()
    plt.bar(metrics, values)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'transfer_learning_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 실행 요약 로그 저장
    summary_log = {
        'timestamp': timestamp,
        'config_used': config,
        'data_files': {
            'm001_data': args.m001_data,
            'm014_train_data': args.m014_train_data,
            'm014_test_normal': args.m014_test_normal,
            'm014_test_anomaly': args.m014_test_anomaly
        },
        'model_performance': performance_df.to_dict(),
        'final_threshold': float(threshold),
        'data_statistics': {
            'train_sequences': len(train_sequences),
            'test_normal_sequences': len(test_normal_sequences),
            'test_anomaly_sequences': len(test_anomaly_sequences)
        }
    }
    
    with open(os.path.join(logs_dir, 'experiment_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_log, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Transfer Learning 완료!")
    print("="*80)
    print(f"결과 폴더: {transfer_run_dir}")
    print(f"  ├── models/transfer_model.pth")
    print(f"  ├── results/performance_metrics.csv")
    print(f"  ├── plots/transfer_learning_summary.png")
    print(f"  └── logs/experiment_summary.json")
    print("="*80)
    
    print(f"\n결과가 '{paths_config['results_dir']}' 폴더에 저장되었습니다.")
    print("Transfer Learning 완료!")

if __name__ == "__main__":
    main()