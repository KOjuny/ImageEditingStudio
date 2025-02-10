import pandas as pd

# CSV 파일 읽기
file_path = '/home/poong/junseok/PnPInversion/evaluation/masactrl_results.csv'  # 여기에 실제 파일 경로를 입력하세요
data = pd.read_csv(file_path)

# 각 열의 평균 계산
column_means = data.mean()

for col in column_means.index:
    if 'structure_distance' in col:
        column_means[col] *= 1000
    elif "lpips" in col:
        column_means[col] *= 1000
    elif "mse" in col:
        column_means[col] *= 10000
    elif "ssim" in col:
        column_means[col] *= 100

# 평균 값을 소수점 둘째 자리까지 반올림
column_means_new = column_means.round(2)

# 결과를 DataFrame으로 변환 (열 이름 유지)
means_df = pd.DataFrame(column_means_new).T  # 전치(transpose)하여 열 이름을 유지
means_df.index = ['Mean']  # 행 이름 설정

# 결과를 CSV 파일로 저장
output_path = 'average_masactrl.csv'  # 저장할 파일 경로
means_df.to_csv(output_path, index=False)

# 결과 출력
print("각 열의 평균:")
print(means_df)
