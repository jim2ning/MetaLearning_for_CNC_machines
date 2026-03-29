import sys
import io
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import create_meta_tasks, create_finetune_data, load_and_preprocess_data
from networks import LSTMVAE
import numpy as np
from tqdm import tqdm
import psutil
from psutil import Process
import time
import torch.nn.functional as F
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import StepLR

# 한글 출력을 위한 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# GPU 사용 설정 강화
if torch.cuda.is_available():
    device = torch.device('cuda')
    # GPU 메모리 캐시 초기화
    torch.cuda.empty_cache()
    # GPU 사용 가능 여부 확인
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    # 기본 GPU 설정
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")

print(f"Using device: {device}")

def meta_train(model, meta_optimizer, train_data, val_data, meta_epochs=300, tasks_per_meta_update=4, inner_steps=5, inner_lr=0.01, batch_size=16, start_epoch=0):
    """메타 학습 (MAML) 훈련 함수"""
    # 학습률 스케줄러 정의
    scheduler = StepLR(meta_optimizer, step_size=50, gamma=0.5)
    os.makedirs('trained_model', exist_ok=True)
    print("\n메타 학습 시작...")
    for epoch in range(start_epoch, meta_epochs):
        # GPU 메모리 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        task_losses = []

        for task_idx in range(len(train_data)):
            try:
                print(f"\nTask {task_idx+1}/{len(train_data)}")
                
                # # Task 5 디버깅을 위한 추가 로그
                # if task_idx == 4:  # Task 5 (0-based index)
                #     print("\n=== Task 5 디버깅 정보 ===")
                #     print(f"학습 데이터 크기: {len(train_data[task_idx])}")
                #     print(f"검증 데이터 크기: {len(val_data[task_idx])}")
                #     if len(train_data[task_idx]) > 0:
                #         print(f"첫 번째 학습 시퀀스 크기: {train_data[task_idx][0].shape}")
                #     if len(val_data[task_idx]) > 0:
                #         print(f"첫 번째 검증 시퀀스 크기: {val_data[task_idx][0].shape}")
                #     print("========================\n")
                
                # 각 태스크 시작 시 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
                
                # 내부 루프 학습
                inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr, weight_decay=1e-5)
                
                # 시퀀스 데이터를 배치로 구성
                train_sequences = train_data[task_idx]
                test_sequences = val_data[task_idx]
                
                # 학습 진행률 표시
                pbar = tqdm(range(inner_steps), desc=f'Inner Training')
                for inner_step in pbar:
                    epoch_loss = 0
                    batch_count = 0
                    
                    # 시퀀스를 배치로 구성
                    for i in range(0, len(train_sequences), batch_size):
                        try:
                            batch_sequences = train_sequences[i:i + batch_size]
                            if not batch_sequences:
                                continue
                            
                            # 배치 처리 전 메모리 정리
                            if batch_count % 5 == 0:  # 더 자주 메모리 정리
                                torch.cuda.empty_cache()
                                gc.collect()
                                
                            # 배치 내의 모든 시퀀스를 동일한 길이로 패딩
                            max_len = max(seq.size(0) for seq in batch_sequences)
                            padded_sequences = []
                            for seq in batch_sequences:
                                if seq.size(0) < max_len:
                                    padding = seq[-1].unsqueeze(0).repeat(max_len - seq.size(0), 1)
                                    seq = torch.cat([seq, padding], dim=0)
                                padded_sequences.append(seq)
                            
                            batch = torch.stack(padded_sequences).to(device)
                            
                            inner_optimizer.zero_grad()
                            x_recon, mu, log_var = model(batch)
                            loss = model.compute_loss(batch, x_recon, mu, log_var)
                            
                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f"Warning: Invalid loss value detected: {loss.item()}")
                                continue
                                
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            inner_optimizer.step()
                            
                            epoch_loss += loss.item()
                            batch_count += 1
                            
                            # 메모리 사용량 확인
                            if batch_count % 5 == 0:  # 더 자주 메모리 체크
                                memory = Process().memory_info().rss / 1024 / 1024
                                pbar.set_postfix({
                                    'loss': f'{loss.item():.4f}',
                                    'memory': f'{memory:.1f}MB'
                                })
                            
                            # 배치 처리 후 메모리 정리
                            del batch, x_recon, mu, log_var, loss
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error in batch processing: {str(e)}")
                            continue
                    
                    if batch_count > 0:
                        pbar.set_description(f'Inner Epoch {inner_step+1}/{inner_steps} (Loss: {epoch_loss/batch_count:.4f})')
                
                # 메타 학습
                task_loss = torch.tensor(0.0, requires_grad=True)
                test_batch_count = 0
                valid_losses = []  # 유효한 손실값만 저장
                
                for i in range(0, len(test_sequences), batch_size):
                    try:
                        batch_sequences = test_sequences[i:i + batch_size]
                        if not batch_sequences:
                            continue
                            
                        # 배치 내의 모든 시퀀스를 동일한 길이로 패딩
                        max_len = max(seq.size(0) for seq in batch_sequences)
                        padded_sequences = []
                        for seq in batch_sequences:
                            if seq.size(0) < max_len:
                                padding = seq[-1].unsqueeze(0).repeat(max_len - seq.size(0), 1)
                                seq = torch.cat([seq, padding], dim=0)
                            padded_sequences.append(seq)
                        
                        batch = torch.stack(padded_sequences).to(device)
                        
                        x_recon, mu, log_var = model(batch)
                        loss = model.compute_loss(batch, x_recon, mu, log_var)
                        
                        # 유효한 손실값인 경우에만 저장
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            valid_losses.append(loss.item())
                            task_loss = task_loss + loss
                            test_batch_count += 1
                        
                        # 배치 처리 후 메모리 정리
                        del batch, x_recon, mu, log_var, loss
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error in test batch processing: {str(e)}")
                        continue
                
                if test_batch_count > 0 and valid_losses:  # 유효한 손실값이 있는 경우에만 처리
                    task_loss = task_loss / test_batch_count
                    task_losses.append(task_loss.item())
                    print(f'Task {task_idx+1} Test Loss: {task_loss.item():.4f}')
                    
                    # 각 task 완료 후 모델 저장 (가장 최근 상태만 유지)
                    torch.save({
                        'task_idx': task_idx,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                        'task_loss': task_loss.item(),
                        'meta_loss': epoch_loss / batch_count if batch_count > 0 else float('inf'),
                        'hyperparameters': {
                            'inner_steps': inner_steps,
                            'inner_lr': inner_lr,
                            'batch_size': batch_size
                        }
                    }, os.path.join('trained_model', f'meta_vae_model_task_{task_idx+1}_latest.pth'))
                    print(f"Task {task_idx+1} 모델 저장됨 (Epoch {epoch+1})")
                else:
                    print(f'Task {task_idx+1}: No valid test losses recorded')
            
            except Exception as e:
                print(f"Error in Task {task_idx+1}: {str(e)}")
                # 오류 발생 시에도 현재 상태 저장
                torch.save({
                    'task_idx': task_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                    'error': str(e),
                    'hyperparameters': {
                        'inner_steps': inner_steps,
                        'inner_lr': inner_lr,
                        'batch_size': batch_size
                    }
                }, os.path.join('trained_model', f'meta_vae_model_task_{task_idx+1}_latest.pth'))
                print(f"Task {task_idx+1} 오류 발생, 현재 상태 저장됨 (Epoch {epoch+1})")
                continue
        
        # 외부 루프: 메타 옵티마이저로 모델 파라미터 업데이트
        meta_optimizer.step()

        # GPU 메모리 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 학습률 스케줄러 업데이트
        scheduler.step()

        # 에포크 결과 출력
        if task_losses:  # task_losses가 비어있지 않은 경우에만 평균 계산
            avg_loss = np.mean(task_losses)
            print(f'Epoch {epoch+1}/{meta_epochs}, Average Meta Loss: {avg_loss:.4f}')
            
            # 모든 task의 학습이 끝난 후 각 epoch가 완료될 때마다 모델 저장
            print(f"\nEpoch {epoch+1} 완료 - 메타러닝 모델 저장 중...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                'meta_epochs': meta_epochs,
                'tasks_per_meta_update': tasks_per_meta_update,
                'inner_steps': inner_steps,
                'inner_lr': inner_lr,
                'batch_size': batch_size,
                'meta_lr': meta_optimizer.param_groups[0]['lr'],
                'avg_loss': avg_loss,
                'task_losses': task_losses
            }, os.path.join('trained_model', f'meta_vae_model_epoch_{epoch+1}.pth'))
            print(f"Epoch {epoch+1} 메타러닝 모델이 'meta_vae_model_epoch_{epoch+1}.pth'로 저장되었습니다.")
        else:
            print(f'Epoch {epoch+1}/{meta_epochs}, No valid task losses recorded')

    print("\n메타 학습 완료.")
    
    # 모든 task의 학습이 완료된 후 최종 메타러닝 모델 저장
    print("\n최종 메타러닝 모델 저장 중...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'meta_optimizer_state_dict': meta_optimizer.state_dict(),
        'meta_epochs': meta_epochs,
        'tasks_per_meta_update': tasks_per_meta_update,
        'inner_steps': inner_steps,
        'inner_lr': inner_lr,
        'batch_size': batch_size,
        'meta_lr': meta_optimizer.param_groups[0]['lr'],
        'final_avg_loss': np.mean(task_losses) if task_losses else float('inf')
    }, os.path.join('trained_model', 'meta_vae_model_completed.pth'))
    print("최종 메타러닝 모델이 'meta_vae_model_completed.pth'로 저장되었습니다.")
    
    return model

def finetune(model, train_sequences, test_sequences, epochs=100, batch_size=64, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        # 학습
        model.train()
        train_loss = 0
        batch_count = 0
        start_time = time.time()
        
        pbar = tqdm(range(0, len(train_sequences), batch_size), desc=f'Training')
        for i in pbar:
            batch_sequences = train_sequences[i:i + batch_size]
            if not batch_sequences:
                continue
                
            # 배치 내의 모든 시퀀스를 동일한 길이로 패딩
            max_len = max(seq.size(0) for seq in batch_sequences)
            padded_sequences = []
            for seq in batch_sequences:
                if seq.size(0) < max_len:
                    padding = torch.zeros(max_len - seq.size(0), seq.size(1))
                    seq = torch.cat([seq, padding], dim=0)
                padded_sequences.append(seq)
            
            batch = torch.stack(padded_sequences).to(device)
            
            optimizer.zero_grad()
            x_recon, mu, log_var = model(batch)
            loss = model.compute_loss(batch, x_recon, mu, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                memory = Process().memory_info().rss / 1024 / 1024
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'memory': f'{memory:.1f}MB'
                })
            
            # GPU 메모리 관리
            if batch_count % 5 == 0:
                torch.cuda.empty_cache()
        
        # 평가
        model.eval()
        test_loss = 0
        test_batch_count = 0
        anomalies = []
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_sequences = test_sequences[i:i + batch_size]
                if not batch_sequences:
                    continue
                    
                # 배치 내의 모든 시퀀스를 동일한 길이로 패딩
                max_len = max(seq.size(0) for seq in batch_sequences)
                padded_sequences = []
                for seq in batch_sequences:
                    if seq.size(0) < max_len:
                        padding = torch.zeros(max_len - seq.size(0), seq.size(1))
                        seq = torch.cat([seq, padding], dim=0)
                    padded_sequences.append(seq)
                
                batch = torch.stack(padded_sequences).to(device)
                
                x_recon, mu, log_var = model(batch)
                loss = model.compute_loss(batch, x_recon, mu, log_var)
                test_loss += loss.item()
                test_batch_count += 1
                
                # 이상 탐지
                reconstruction_error = F.mse_loss(x_recon, batch, reduction='none')
                mean_error = reconstruction_error.mean(dim=(1, 2))
                anomalies.extend(mean_error.cpu().numpy())
                
                # GPU 메모리 관리
                if test_batch_count % 5 == 0:
                    torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1} Summary:')
        if batch_count > 0:
            print(f'Train Loss: {train_loss/batch_count:.4f}')
        if test_batch_count > 0:
            print(f'Test Loss: {test_loss/test_batch_count:.4f}')
        print(f'Anomaly Detection Rate: {np.mean(anomalies):.4f}')
        print(f'Time Elapsed: {elapsed_time:.2f}s')
        print(f'Memory Usage: {Process().memory_info().rss / 1024 / 1024:.1f}MB')

def load_model(model_path, model):
    """저장된 모델을 불러오는 함수"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"모델이 {device}로 로드되었습니다.")
        return model, checkpoint
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None

def finetune_model(model, target_machine, finetune_epochs, finetune_batch_size, finetune_lr, meta_epochs, tasks_per_meta_update, inner_steps, inner_lr, batch_size, meta_lr, hidden_dim, latent_dim, num_layers, dropout_rate):
    
    # 파인튜닝 데이터 준비
    print(f"\n머신 {target_machine} 파인튜닝 준비...")
    train_sequences, test_sequences, _ = create_finetune_data(target_machine)
    print("파인튜닝 데이터 준비 완료.")
    
    # 파인튜닝
    print("\n파인튜닝 시작...")
    finetune(model, train_sequences, test_sequences,
            epochs=finetune_epochs, batch_size=finetune_batch_size, lr=finetune_lr)
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'target_machine': target_machine,
        'hyperparameters': {
            'meta_epochs': meta_epochs,
            'tasks_per_meta_update': tasks_per_meta_update,
            'inner_steps': inner_steps,
            'inner_lr': inner_lr,
            'batch_size': batch_size,
            'finetune_epochs': finetune_epochs,
            'finetune_batch_size': finetune_batch_size,
            'finetune_lr': finetune_lr,
            'meta_lr': meta_lr,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate
        }
    }, os.path.join('trained_model', 'final_meta_vae_model.pth'))
    
    print("\n학습 완료!")

def main():
    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 하이퍼파라미터 설정
    print("\n하이퍼파라미터 설정")
    print("1. 기본값 사용")
    print("2. 사용자 정의")
    choice = input("선택하세요 (1/2): ")
    
    if choice == "2":
        print("\n=== 메타 학습 하이퍼파라미터 ===")
        meta_epochs = int(input("메타 에포크 수 (기본값: 300): ") or "300")
        tasks_per_meta_update = int(input("각 에포크에서 학습할 태스크 수 (기본값: 4): ") or "4")
        inner_steps = int(input("내부 에포크 수 (기본값: 5): ") or "5")
        inner_lr = float(input("내부 학습률 (기본값: 0.01): ") or "0.01")
        batch_size = int(input("메타 배치 크기 (기본값: 16): ") or "16")
        meta_lr = float(input("메타 학습률 (기본값: 0.0005): ") or "0.0005")
        
        print("\n=== 모델 구조 하이퍼파라미터 ===")
        hidden_dim = int(input("은닉층 차원 (기본값: 64): ") or "64")
        latent_dim = int(input("잠재 공간 차원 (기본값: 32): ") or "32")
        num_layers = int(input("LSTM 레이어 수 (기본값: 2): ") or "2")
        dropout_rate = float(input("드롭아웃 비율 (기본값: 0.1): ") or "0.1")
        
        print("\n=== 파인튜닝 하이퍼파라미터 ===")
        finetune_epochs = int(input("파인튜닝 에포크 수 (기본값: 100): ") or "100")
        finetune_batch_size = int(input("파인튜닝 배치 크기 (기본값: 64): ") or "64")
        finetune_lr = float(input("파인튜닝 학습률 (기본값: 0.0005): ") or "0.0005")
        
        print("\n=== 데이터 전처리 하이퍼파라미터 ===")
        sequence_length = int(input("시퀀스 길이 (기본값: 100): ") or "100")
        overlap = int(input("시퀀스 중첩 길이 (기본값: 50): ") or "50")
        normalize = input("데이터 정규화 사용 (y/n, 기본값: y): ").lower() == 'y'
        use_augmentation = input("데이터 증강 사용 (y/n, 기본값: n): ").lower() == 'y'
    else:
        # 기본값 설정
        meta_epochs = 300
        tasks_per_meta_update = 4
        inner_steps = 5
        inner_lr = 0.01
        batch_size = 16
        meta_lr = 0.0005
        
        hidden_dim = 64
        latent_dim = 32
        num_layers = 2
        dropout_rate = 0.1
        
        finetune_epochs = 100
        finetune_batch_size = 64
        finetune_lr = 0.0005
        
        sequence_length = 100
        overlap = 50
        normalize = True
        use_augmentation = False
    
    # 메타 학습에 사용할 머신 ID 목록
    meta_machines = [1, 2, 5, 6, 13, 14]
    
    # 메타 학습 태스크 생성
    print("\n메타 학습 태스크 생성 중...")
    train_tasks, test_tasks = create_meta_tasks(
        meta_machines,
        sequence_length=sequence_length,
        overlap=overlap,
        normalize=normalize,
        use_augmentation=use_augmentation
    )
    print("메타 학습 태스크 생성 완료.")
    
    # 메타 학습 모델 초기화
    model = LSTMVAE(
        input_dim=5,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)
    
    # 메타 옵티마이저 초기화
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr, weight_decay=1e-5)
    
    # 이전에 저장된 모델이 있는지 확인
    last_saved_task = None
    for task_idx in range(len(meta_machines)-1, -1, -1):
        model_path = f'meta_vae_model_task_{task_idx+1}_latest.pth'
        if os.path.exists(model_path):
            print(f"\n이전 학습 모델 발견: Task {task_idx+1}")
            choice = input("이전 모델을 불러와서 계속 학습하시겠습니까? (y/n): ").lower()
            
            if choice == 'y':
                model, checkpoint = load_model(model_path, model)
                if model is not None and checkpoint and 'meta_optimizer_state_dict' in checkpoint:
                    try:
                        meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
                        print("메타 옵티마이저 상태를 불러왔습니다.")
                    except Exception as e:
                        print(f"메타 옵티마이저 상태 로드 중 오류 발생: {str(e)}")
                        print("새로운 메타 옵티마이저로 계속 진행합니다.")
                elif model is not None:
                    print("경고: 저장된 체크포인트에 메타 옵티마이저 상태가 없습니다. 새로운 메타 옵티마이저로 계속 진행합니다.")
                
                if model is not None:
                    last_saved_task = task_idx
                    print(f"Task {task_idx+1} 모델을 불러왔습니다.")
                break
            else:
                print("처음부터 새로 학습을 시작합니다.")
                break
    
    # 메타 학습 시작
    print("\n메타 학습 시작...")
    if last_saved_task is not None:
        print(f"Task {last_saved_task + 1}부터 학습을 재개합니다...")
    # ----------------- 메타러닝 학습 시작 -----------------
    model = meta_train(model, meta_optimizer, train_tasks, test_tasks,
                      meta_epochs=meta_epochs, tasks_per_meta_update=tasks_per_meta_update,
                      inner_steps=inner_steps, inner_lr=inner_lr, batch_size=batch_size,
                      start_epoch=0)  # task 기반으로 저장되므로 epoch는 0부터 시작
    # ----------------- 메타러닝 학습 완료 -----------------
    
    # ----------------- 파인튜닝 시작 -----------------
    # 파인튜닝할 머신 선택
    target_machine = 14
    finetune_model(model, target_machine, finetune_epochs, finetune_batch_size, finetune_lr, meta_epochs, tasks_per_meta_update, inner_steps, inner_lr, batch_size, meta_lr, hidden_dim, latent_dim, num_layers, dropout_rate)
    # ----------------- 파인튜닝 완료 -----------------


if __name__ == "__main__":
    main()
