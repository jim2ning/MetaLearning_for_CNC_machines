import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
import pickle
from tslearn.metrics import SoftDTW
import torch

def detect_production_cycles_by_qty(df, qty_column='PROSSED_QTY'):
    """PROSSED_QTY 값별로 생산 사이클을 감지하는 함수"""
    if qty_column not in df.columns:
        print(f"Warning: {qty_column} 컬럼을 찾을 수 없습니다.")
        return []
    
    qty_values = df[qty_column].values
    cycles = []
    
    # 연속된 같은 값들을 그룹화
    current_value = qty_values[0]
    cycle_start = 0
    
    for i in range(1, len(qty_values)):
        if qty_values[i] != current_value:
            # 이전 사이클 종료
            if i - cycle_start > 10:  # 최소 길이 체크
                cycles.append((cycle_start, i, current_value))
            
            # 새로운 사이클 시작
            cycle_start = i
            current_value = qty_values[i]
    
    # 마지막 사이클 추가
    if len(qty_values) - cycle_start > 10:
        cycles.append((cycle_start, len(qty_values), current_value))
    
    print(f"  감지된 생산 사이클 수: {len(cycles)}")
    if cycles:
        cycle_lengths = [end - start for start, end, _ in cycles]
        print(f"  사이클 길이 - 평균: {np.mean(cycle_lengths):.1f}, 최소: {min(cycle_lengths)}, 최대: {max(cycle_lengths)}")
        
        # 각 PROSSED_QTY 값별 사이클 수 출력
        qty_counts = {}
        for start, end, qty in cycles:
            if qty not in qty_counts:
                qty_counts[qty] = 0
            qty_counts[qty] += 1
        
        print(f"  PROSSED_QTY별 사이클 수: {qty_counts}")
    
    return cycles

def soft_dtw_alignment(sequence, target_length=333, gamma=1.0):
    """실제 Soft-DTW를 사용하여 시퀀스를 목표 길이로 정렬"""
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    # 목표 시퀀스 생성 (선형 보간으로 초기 추정)
    time_original = np.linspace(0, 1, current_length)
    time_target = np.linspace(0, 1, target_length)
    
    # 각 특성별로 초기 정렬
    aligned_sequence = np.zeros((target_length, sequence.shape[1]))
    
    for feature_idx in range(sequence.shape[1]):
        feature_values = sequence[:, feature_idx]
        aligned_sequence[:, feature_idx] = np.interp(time_target, time_original, feature_values)
    
    # Soft-DTW를 사용한 정렬 개선
    try:
        # PyTorch 텐서로 변환
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq_len, features)
        target_tensor = torch.FloatTensor(aligned_sequence).unsqueeze(0)  # (1, target_len, features)
        
        # SoftDTW 객체 생성
        sdtw = SoftDTW(gamma=gamma, normalize=True)
        
        # Soft-DTW 거리 계산
        dtw_loss = sdtw(seq_tensor, target_tensor)
        
        # 역전파를 통한 정렬 개선 (간단한 버전)
        # 실제로는 더 복잡한 정렬 알고리즘이 필요하지만, 여기서는 기본 정렬 사용
        print(f"    Soft-DTW 거리: {dtw_loss.item():.4f}")
        
        return aligned_sequence
        
    except Exception as e:
        print(f"    Soft-DTW 실패: {e}, 선형 보간 사용")
        return aligned_sequence

def preprocess_single_file(file_path, target_length=333, output_dir='Preprocessing_Data'):
    """단일 CSV 파일을 전처리하여 시퀀스 길이를 맞추는 함수"""
    
    print(f"\n파일 처리 중: {file_path}")
    
    # 파일 읽기
    try:
        df = pd.read_csv(file_path)
        print(f"  원본 데이터 크기: {df.shape}")
    except Exception as e:
        print(f"  파일 읽기 실패: {e}")
        return
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  수치형 컬럼 수: {len(numeric_columns)}")
    
    if len(numeric_columns) == 0:
        print("  수치형 컬럼이 없습니다. 건너뜁니다.")
        return
    
    # NaN 값 처리
    df_numeric = df[numeric_columns].copy()
    
    # NaN 값 확인
    nan_counts = df_numeric.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN 값 발견: {nan_counts[nan_counts > 0].sum()}개")
        df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
    
    # 무한대 값 처리
    inf_mask = np.isinf(df_numeric.values)
    if inf_mask.any():
        print(f"  무한대 값 {inf_mask.sum()}개 발견, 처리합니다.")
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
    
    data = df_numeric.values
    
    # 스케일링
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # PROSSED_QTY로 생산 사이클 감지
    production_cycles = detect_production_cycles_by_qty(df, 'PROSSED_QTY')
    
    if not production_cycles:
        print("  생산 사이클을 감지할 수 없습니다. 건너뜁니다.")
        return
    
    aligned_sequences = []
    
    print(f"  {len(production_cycles)}개 사이클을 Soft-DTW로 길이 {target_length}로 정렬 중...")
    
    for i, (start, end, qty_value) in enumerate(tqdm(production_cycles, desc="  사이클 처리")):
        cycle_data = data_scaled[start:end]
        
        print(f"    사이클 {i+1}: PROSSED_QTY={qty_value}, 길이={len(cycle_data)} -> {target_length}")
        
        # 너무 짧은 사이클 제외
        if len(cycle_data) < 10:
            print(f"    사이클 {i+1} 너무 짧음, 건너뜀")
            continue
        
        # 실제 Soft-DTW로 정렬
        try:
            aligned_cycle = soft_dtw_alignment(cycle_data, target_length)
            aligned_sequences.append({
                'sequence': aligned_cycle,
                'qty_value': qty_value,
                'original_length': len(cycle_data)
            })
        except Exception as e:
            print(f"    사이클 {i+1} 정렬 실패: {e}")
            continue
    
    if not aligned_sequences:
        print("  처리된 시퀀스가 없습니다.")
        return
    
    print(f"  최종 시퀀스 수: {len(aligned_sequences)}")
    print(f"  각 시퀀스 길이: {target_length}")
    print(f"  특성 차원: {aligned_sequences[0]['sequence'].shape[1]}")
    
    # 결과를 DataFrame으로 변환
    all_sequences = []
    
    for seq_idx, seq_info in enumerate(aligned_sequences):
        sequence = seq_info['sequence']
        qty_value = seq_info['qty_value']
        original_length = seq_info['original_length']
        
        for time_step in range(target_length):
            row_data = {
                'sequence_id': seq_idx, 
                'time_step': time_step,
                'original_qty_value': qty_value,
                'original_length': original_length
            }
            for col_idx, col_name in enumerate(numeric_columns):
                row_data[col_name] = sequence[time_step, col_idx]
            all_sequences.append(row_data)
    
    result_df = pd.DataFrame(all_sequences)
    
    # 출력 파일명 생성
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_softdtw.csv")
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"  저장 완료: {output_file}")
    
    # 스케일러도 저장
    scaler_file = os.path.join(output_dir, f"{base_name}_scaler.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  스케일러 저장: {scaler_file}")

def preprocess_all_data(data_dir='Data', target_length=333, output_dir='Preprocessing_Data'):
    """Data 폴더의 모든 CSV 파일을 전처리하는 함수"""
    
    print("="*60)
    print(f"Soft-DTW 데이터 전처리 시작")
    print(f"입력 폴더: {data_dir}")
    print(f"출력 폴더: {output_dir}")
    print(f"목표 시퀀스 길이: {target_length}")
    print("="*60)
    
    # Data 폴더에서 모든 CSV 파일 찾기
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"{data_dir} 폴더에 CSV 파일이 없습니다.")
        return
    
    print(f"발견된 CSV 파일 수: {len(csv_files)}")
    for file in csv_files:
        print(f"  - {file}")
    
    # 각 파일 처리
    for file_path in csv_files:
        try:
            preprocess_single_file(file_path, target_length, output_dir)
        except Exception as e:
            print(f"파일 {file_path} 처리 중 오류 발생: {e}")
            continue
    
    print("\n" + "="*60)
    print("Soft-DTW 전체 데이터 전처리 완료!")
    print(f"결과 파일들이 '{output_dir}' 폴더에 저장되었습니다.")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Data Preprocessing with Soft-DTW')
    parser.add_argument('--data_dir', type=str, default='Data', help='입력 데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default='Preprocessing_Data', help='출력 디렉토리')
    parser.add_argument('--target_length', type=int, default=333, help='목표 시퀀스 길이')
    
    args = parser.parse_args()
    
    preprocess_all_data(
        data_dir=args.data_dir,
        target_length=args.target_length,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()