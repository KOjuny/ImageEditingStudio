import pandas as pd


file_path = ''  
data = pd.read_csv(file_path)


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


column_means_new = column_means.round(2)


means_df = pd.DataFrame(column_means_new).T  
means_df.index = ['Mean'] 


output_path = 'average_masactrl.csv'  
means_df.to_csv(output_path, index=False)

print("각 열의 평균:")
print(means_df)
