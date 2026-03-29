import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import io
import os

# 한글 출력을 위한 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length + 1
        
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length]

def load_and_preprocess_data(machine_id, data_type='normal', seq_length=None):
    """각 머신의 데이터를 로드하고 전처리합니다."""
    # 데이터 파일 경로
    file_path = os.path.join('Data', f'M{machine_id:03d}_{data_type}_processed.csv')
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"경고: 파일이 존재하지 않습니다: {file_path}")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        return [], None, None, []
    
    try:
        # 데이터 로드 (모든 컬럼)
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"경고: 파일이 비어있습니다: {file_path}")
            return [], None, None, []
        
        # PROCESSD_QTY의 변화 지점 찾기
        qty_changes = df['PROSSED_QTY'].diff().ne(0).cumsum()
        sequences = []
        
        # 각 변화 지점별로 시퀀스 생성
        for _, group in df.groupby(qty_changes):
            if len(group) < 2:  # 너무 짧은 시퀀스는 건너뛰기
                continue
                
            # LOAD_1~5 컬럼만 선택
            load_columns = [f'LOAD_{i}' for i in range(1, 6)]
            sequence = group[load_columns].values.astype(np.float32)
            sequences.append(sequence)
        
        print(f"\n머신 {machine_id}의 {data_type} 데이터 분석:")
        print(f"총 시퀀스 개수: {len(sequences)}")
        if sequences:
            print(f"시퀀스 길이 분포: {[len(seq) for seq in sequences]}")
            print(f"평균 시퀀스 길이: {np.mean([len(seq) for seq in sequences]):.2f}")
            print(f"최소 시퀀스 길이: {min([len(seq) for seq in sequences])}")
            print(f"최대 시퀀스 길이: {max([len(seq) for seq in sequences])}")
        
        # 시퀀스를 텐서로 변환
        sequences = [torch.FloatTensor(seq) for seq in sequences]
        
        if not sequences:
            return [], None, None, []
        
        # 데이터 정규화
        all_data = torch.cat(sequences, dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        
        normalized_sequences = [(seq - mean) / (std + 1e-8) for seq in sequences]
        
        return normalized_sequences, mean, std, load_columns
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        return [], None, None, []

def create_sequences(data, sequence_length, overlap):
    """시퀀스 데이터를 생성합니다."""
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, sequence_length - overlap):
        sequence = data[i:i + sequence_length]
        if len(sequence) == sequence_length:
            sequences.append(sequence)
    return np.array(sequences)

def create_meta_tasks(machine_ids, sequence_length=100, overlap=50, normalize=True, use_augmentation=False):
    """메타 학습을 위한 태스크 생성"""
    train_tasks = []
    test_tasks = []
    
    for machine_id in machine_ids:
        # 정상 데이터와 고장 데이터 로드
        normal_sequences, normal_mean, normal_std, normal_cols = load_and_preprocess_data(machine_id, 'normal')
        faulty_sequences, faulty_mean, faulty_std, faulty_cols = load_and_preprocess_data(machine_id, 'faulty')
        
        # 공통 컬럼 찾기
        common_cols = list(set(normal_cols) & set(faulty_cols))
        normal_indices = [normal_cols.index(col) for col in common_cols]
        faulty_indices = [faulty_cols.index(col) for col in common_cols]
        
        # 공통 컬럼만 선택
        normal_sequences = [seq[:, normal_indices] for seq in normal_sequences]
        faulty_sequences = [seq[:, faulty_indices] for seq in faulty_sequences]
        
        # 모든 시퀀스를 하나의 리스트로 합치기
        all_sequences = normal_sequences + faulty_sequences
        
        if use_augmentation:
            # 데이터 증강
            augmented_sequences = []
            for seq in all_sequences:
                # 노이즈 추가
                noise = torch.randn_like(seq) * 0.01
                augmented_sequences.append(seq + noise)
                # 시간 이동
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    shifted_seq = torch.roll(seq, shift, dims=0)
                    augmented_sequences.append(shifted_seq)
            all_sequences.extend(augmented_sequences)
        
        # 학습/테스트 분할
        train_size = int(0.8 * len(all_sequences))
        train_sequences = all_sequences[:train_size]
        test_sequences = all_sequences[train_size:]
        
        train_tasks.append(train_sequences)
        test_tasks.append(test_sequences)
    
    return train_tasks, test_tasks

def create_finetune_data(machine_id, seq_length=None):
    """파인튜닝을 위한 데이터를 생성합니다."""
    print(f"\n머신 {machine_id} 파인튜닝 데이터 준비 중...")
    
    # 정상 데이터와 고장 데이터 로드
    normal_sequences, normal_mean, normal_std, normal_cols = load_and_preprocess_data(machine_id, 'normal')
    faulty_sequences, faulty_mean, faulty_std, faulty_cols = load_and_preprocess_data(machine_id, 'faulty')
    
    # 공통 컬럼 찾기
    common_cols = list(set(normal_cols) & set(faulty_cols))
    normal_indices = [normal_cols.index(col) for col in common_cols]
    faulty_indices = [faulty_cols.index(col) for col in common_cols]
    
    # 공통 컬럼만 선택
    normal_sequences = [seq[:, normal_indices] for seq in normal_sequences]
    faulty_sequences = [seq[:, faulty_indices] for seq in faulty_sequences]
    
    # 데이터 분할 (80% 학습, 20% 테스트)
    normal_train_size = int(0.8 * len(normal_sequences))
    faulty_train_size = int(0.8 * len(faulty_sequences))
    
    # 학습 데이터
    train_sequences = normal_sequences[:normal_train_size]
    
    # 테스트 데이터
    test_sequences = normal_sequences[normal_train_size:]
    
    # 이상 데이터
    anomaly_sequences = faulty_sequences
    
    print(f"학습 시퀀스 수: {len(train_sequences)}")
    print(f"테스트 시퀀스 수: {len(test_sequences)}")
    
    return train_sequences, test_sequences, anomaly_sequences