import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_machine_data(machine_id):
    """특정 기계의 normal과 faulty 데이터를 로드하고 분석"""
    
    print(f"\n=== {machine_id} 데이터 분석 ===")
    
    # 파일 경로
    normal_file = f'Preprocessing_Data/{machine_id}_normal_processed_softdtw.csv'
    faulty_file = f'Preprocessing_Data/{machine_id}_faulty_processed_softdtw.csv'
    
    try:
        # 데이터 로드
        normal_df = pd.read_csv(normal_file)
        faulty_df = pd.read_csv(faulty_file)
        
        print(f"Normal 데이터: {normal_df.shape}")
        print(f"Faulty 데이터: {faulty_df.shape}")
        
        # LOAD 컬럼들
        load_columns = ['LOAD_1', 'LOAD_2', 'LOAD_3', 'LOAD_4', 'LOAD_5']
        available_loads = [col for col in load_columns if col in normal_df.columns]
        
        # 기본 통계
        normal_stats = {}
        faulty_stats = {}
        
        for col in available_loads:
            normal_values = normal_df[col].values
            faulty_values = faulty_df[col].values
            
            normal_stats[col] = {
                'mean': normal_values.mean(),
                'std': normal_values.std(),
                'min': normal_values.min(),
                'max': normal_values.max(),
                'median': np.median(normal_values)
            }
            
            faulty_stats[col] = {
                'mean': faulty_values.mean(),
                'std': faulty_values.std(),
                'min': faulty_values.min(),
                'max': faulty_values.max(),
                'median': np.median(faulty_values)
            }
            
            print(f"\n{col}:")
            print(f"  Normal: 평균={normal_stats[col]['mean']:.4f}, 표준편차={normal_stats[col]['std']:.4f}")
            print(f"  Faulty: 평균={faulty_stats[col]['mean']:.4f}, 표준편차={faulty_stats[col]['std']:.4f}")
            print(f"  차이율: {abs(normal_stats[col]['mean'] - faulty_stats[col]['mean']) / normal_stats[col]['mean'] * 100:.2f}%")
        
        return normal_df, faulty_df, normal_stats, faulty_stats, available_loads
        
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None, None, None, None, None

def plot_load_distributions_comparison(machines=['M014', 'M013']):
    """기계별 LOAD 분포 비교 시각화"""
    
    fig, axes = plt.subplots(len(machines), 5, figsize=(25, 6*len(machines)))
    if len(machines) == 1:
        axes = axes.reshape(1, -1)
    
    machine_data = {}
    
    for i, machine_id in enumerate(machines):
        print(f"\n{'='*50}")
        print(f"{machine_id} 분석 중...")
        print(f"{'='*50}")
        
        normal_df, faulty_df, normal_stats, faulty_stats, available_loads = load_and_analyze_machine_data(machine_id)
        
        if normal_df is None:
            continue
            
        machine_data[machine_id] = {
            'normal_df': normal_df,
            'faulty_df': faulty_df,
            'normal_stats': normal_stats,
            'faulty_stats': faulty_stats,
            'available_loads': available_loads
        }
        
        # 각 LOAD별 분포 시각화
        for j, load_col in enumerate(available_loads):
            ax = axes[i, j]
            
            normal_values = normal_df[load_col].values
            faulty_values = faulty_df[load_col].values
            
            # 히스토그램
            ax.hist(normal_values, bins=50, alpha=0.7, label='Normal', 
                   color='blue', density=True)
            ax.hist(faulty_values, bins=50, alpha=0.7, label='Faulty', 
                   color='red', density=True)
            
            # 평균선 추가
            ax.axvline(normal_values.mean(), color='blue', linestyle='--', 
                      label=f'Normal Mean: {normal_values.mean():.3f}')
            ax.axvline(faulty_values.mean(), color='red', linestyle='--', 
                      label=f'Faulty Mean: {faulty_values.mean():.3f}')
            
            ax.set_title(f'{machine_id} - {load_col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('load_distributions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return machine_data

def plot_machine_comparison_summary(machine_data):
    """기계간 비교 요약 시각화"""
    
    if len(machine_data) < 2:
        print("비교할 기계가 부족합니다.")
        return
    
    machines = list(machine_data.keys())
    machine1, machine2 = machines[0], machines[1]
    
    load_columns = machine_data[machine1]['available_loads']
    
    # 1. Normal vs Faulty 차이 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 평균값 차이 비교
    ax1 = axes[0, 0]
    normal_diffs_m1 = []
    normal_diffs_m2 = []
    
    for load_col in load_columns:
        # M1 Normal vs Faulty 차이
        normal_mean_m1 = machine_data[machine1]['normal_stats'][load_col]['mean']
        faulty_mean_m1 = machine_data[machine1]['faulty_stats'][load_col]['mean']
        diff_m1 = abs(normal_mean_m1 - faulty_mean_m1) / normal_mean_m1 * 100
        normal_diffs_m1.append(diff_m1)
        
        # M2 Normal vs Faulty 차이
        normal_mean_m2 = machine_data[machine2]['normal_stats'][load_col]['mean']
        faulty_mean_m2 = machine_data[machine2]['faulty_stats'][load_col]['mean']
        diff_m2 = abs(normal_mean_m2 - faulty_mean_m2) / normal_mean_m2 * 100
        normal_diffs_m2.append(diff_m2)
    
    x = np.arange(len(load_columns))
    width = 0.35
    
    ax1.bar(x - width/2, normal_diffs_m1, width, label=machine1, color='skyblue')
    ax1.bar(x + width/2, normal_diffs_m2, width, label=machine2, color='lightcoral')
    ax1.set_title('Normal vs Faulty 평균값 차이 (%)')
    ax1.set_xlabel('LOAD Columns')
    ax1.set_ylabel('차이율 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(load_columns)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 표준편차 비교
    ax2 = axes[0, 1]
    normal_stds_m1 = [machine_data[machine1]['normal_stats'][col]['std'] for col in load_columns]
    normal_stds_m2 = [machine_data[machine2]['normal_stats'][col]['std'] for col in load_columns]
    
    ax2.bar(x - width/2, normal_stds_m1, width, label=f'{machine1} Normal', color='blue', alpha=0.7)
    ax2.bar(x + width/2, normal_stds_m2, width, label=f'{machine2} Normal', color='red', alpha=0.7)
    ax2.set_title('Normal 데이터 표준편차 비교')
    ax2.set_xlabel('LOAD Columns')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(load_columns)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 범위 비교
    ax3 = axes[0, 2]
    ranges_m1 = [machine_data[machine1]['normal_stats'][col]['max'] - 
                 machine_data[machine1]['normal_stats'][col]['min'] for col in load_columns]
    ranges_m2 = [machine_data[machine2]['normal_stats'][col]['max'] - 
                 machine_data[machine2]['normal_stats'][col]['min'] for col in load_columns]
    
    ax3.bar(x - width/2, ranges_m1, width, label=f'{machine1} Normal', color='blue', alpha=0.7)
    ax3.bar(x + width/2, ranges_m2, width, label=f'{machine2} Normal', color='red', alpha=0.7)
    ax3.set_title('Normal 데이터 범위 비교')
    ax3.set_xlabel('LOAD Columns')
    ax3.set_ylabel('Range (Max - Min)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(load_columns)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 분리도 (Separability) 분석
    ax4 = axes[1, 0]
    separability_m1 = []
    separability_m2 = []
    
    for load_col in load_columns:
        # M1 분리도 (Normal과 Faulty 평균 차이 / 표준편차 합)
        normal_mean_m1 = machine_data[machine1]['normal_stats'][load_col]['mean']
        normal_std_m1 = machine_data[machine1]['normal_stats'][load_col]['std']
        faulty_mean_m1 = machine_data[machine1]['faulty_stats'][load_col]['mean']
        faulty_std_m1 = machine_data[machine1]['faulty_stats'][load_col]['std']
        
        sep_m1 = abs(normal_mean_m1 - faulty_mean_m1) / (normal_std_m1 + faulty_std_m1)
        separability_m1.append(sep_m1)
        
        # M2 분리도
        normal_mean_m2 = machine_data[machine2]['normal_stats'][load_col]['mean']
        normal_std_m2 = machine_data[machine2]['normal_stats'][load_col]['std']
        faulty_mean_m2 = machine_data[machine2]['faulty_stats'][load_col]['mean']
        faulty_std_m2 = machine_data[machine2]['faulty_stats'][load_col]['std']
        
        sep_m2 = abs(normal_mean_m2 - faulty_mean_m2) / (normal_std_m2 + faulty_std_m2)
        separability_m2.append(sep_m2)
    
    ax4.bar(x - width/2, separability_m1, width, label=machine1, color='skyblue')
    ax4.bar(x + width/2, separability_m2, width, label=machine2, color='lightcoral')
    ax4.set_title('Normal-Faulty 분리도 (높을수록 구분하기 쉬움)')
    ax4.set_xlabel('LOAD Columns')
    ax4.set_ylabel('Separability Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(load_columns)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 전체 분리도 요약
    ax5 = axes[1, 1]
    overall_sep_m1 = np.mean(separability_m1)
    overall_sep_m2 = np.mean(separability_m2)
    
    ax5.bar([machine1, machine2], [overall_sep_m1, overall_sep_m2], 
           color=['skyblue', 'lightcoral'])
    ax5.set_title('전체 분리도 비교')
    ax5.set_ylabel('Average Separability Score')
    ax5.grid(True, alpha=0.3)
    
    # 분리도 해석
    ax5.text(0, overall_sep_m1 + 0.01, f'{overall_sep_m1:.3f}', 
            ha='center', va='bottom', fontweight='bold')
    ax5.text(1, overall_sep_m2 + 0.01, f'{overall_sep_m2:.3f}', 
            ha='center', va='bottom', fontweight='bold')
    
    # 데이터 품질 요약
    ax6 = axes[1, 2]
    
    # 변동계수 (CV = std/mean) 비교
    cv_m1 = np.mean([machine_data[machine1]['normal_stats'][col]['std'] / 
                     machine_data[machine1]['normal_stats'][col]['mean'] 
                     for col in load_columns])
    cv_m2 = np.mean([machine_data[machine2]['normal_stats'][col]['std'] / 
                     machine_data[machine2]['normal_stats'][col]['mean'] 
                     for col in load_columns])
    
    ax6.bar([f'{machine1}\n변동계수', f'{machine2}\n변동계수'], [cv_m1, cv_m2], 
           color=['skyblue', 'lightcoral'])
    ax6.set_title('데이터 안정성 비교 (낮을수록 안정)')
    ax6.set_ylabel('Coefficient of Variation')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('machine_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 분석 결과 출력
    print(f"\n{'='*60}")
    print("분석 결과 요약")
    print(f"{'='*60}")
    print(f"{machine1} 전체 분리도: {overall_sep_m1:.4f}")
    print(f"{machine2} 전체 분리도: {overall_sep_m2:.4f}")
    print(f"분리도 차이: {abs(overall_sep_m1 - overall_sep_m2):.4f}")
    
    if overall_sep_m1 > overall_sep_m2:
        print(f"\n✅ {machine1}이 {machine2}보다 Normal-Faulty 구분이 쉬움")
        print(f"   이상 탐지 성능이 더 좋을 것으로 예상됨")
    else:
        print(f"\n✅ {machine2}이 {machine1}보다 Normal-Faulty 구분이 쉬움")
        print(f"   이상 탐지 성능이 더 좋을 것으로 예상됨")
    
    print(f"\n변동계수 비교:")
    print(f"{machine1}: {cv_m1:.4f} (낮을수록 안정)")
    print(f"{machine2}: {cv_m2:.4f} (낮을수록 안정)")

def plot_detailed_load_analysis(machine_data):
    """상세한 LOAD 분석"""
    
    machines = list(machine_data.keys())
    
    for machine_id in machines:
        print(f"\n{'='*50}")
        print(f"{machine_id} 상세 분석")
        print(f"{'='*50}")
        
        data = machine_data[machine_id]
        normal_df = data['normal_df']
        faulty_df = data['faulty_df']
        load_columns = data['available_loads']
        
        # 상관관계 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Normal 데이터 상관관계
        normal_corr = normal_df[load_columns].corr()
        im1 = axes[0, 0].imshow(normal_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 0].set_title(f'{machine_id} - Normal 데이터 상관관계')
        axes[0, 0].set_xticks(range(len(load_columns)))
        axes[0, 0].set_yticks(range(len(load_columns)))
        axes[0, 0].set_xticklabels(load_columns)
        axes[0, 0].set_yticklabels(load_columns)
        
        # 상관계수 값 표시
        for i in range(len(load_columns)):
            for j in range(len(load_columns)):
                axes[0, 0].text(j, i, f'{normal_corr.iloc[i, j]:.2f}', 
                               ha='center', va='center')
        
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Faulty 데이터 상관관계
        faulty_corr = faulty_df[load_columns].corr()
        im2 = axes[0, 1].imshow(faulty_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title(f'{machine_id} - Faulty 데이터 상관관계')
        axes[0, 1].set_xticks(range(len(load_columns)))
        axes[0, 1].set_yticks(range(len(load_columns)))
        axes[0, 1].set_xticklabels(load_columns)
        axes[0, 1].set_yticklabels(load_columns)
        
        for i in range(len(load_columns)):
            for j in range(len(load_columns)):
                axes[0, 1].text(j, i, f'{faulty_corr.iloc[i, j]:.2f}', 
                               ha='center', va='center')
        
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 시계열 패턴 (첫 1000개 샘플)
        axes[1, 0].plot(normal_df[load_columns].iloc[:1000].values, alpha=0.7)
        axes[1, 0].set_title(f'{machine_id} - Normal 시계열 패턴')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('LOAD Values')
        axes[1, 0].legend(load_columns)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Faulty 시계열 패턴
        sample_size = min(1000, len(faulty_df))
        axes[1, 1].plot(faulty_df[load_columns].iloc[:sample_size].values, alpha=0.7)
        axes[1, 1].set_title(f'{machine_id} - Faulty 시계열 패턴')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('LOAD Values')
        axes[1, 1].legend(load_columns)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{machine_id}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 분석 함수"""
    print("LOAD 데이터 분포 분석 시작...")
    print("="*60)
    
    # 기계 리스트 (필요에 따라 수정)
    machines_to_analyze = ['M014', 'M013']
    
    # 1. 기본 분포 비교
    print("\n1. LOAD 분포 비교 시각화...")
    machine_data = plot_load_distributions_comparison(machines_to_analyze)
    
    if len(machine_data) >= 2:
        # 2. 기계간 비교 요약
        print("\n2. 기계간 비교 요약...")
        plot_machine_comparison_summary(machine_data)
        
        # 3. 상세 분석
        print("\n3. 상세 분석...")
        plot_detailed_load_analysis(machine_data)
    
    print("\n분석 완료!")
    print("생성된 파일:")
    print("- load_distributions_comparison.png")
    print("- machine_comparison_summary.png")
    print("- M014_detailed_analysis.png")
    print("- M013_detailed_analysis.png")

if __name__ == "__main__":
    main() 