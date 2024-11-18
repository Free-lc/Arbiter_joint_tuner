# 定义输入和输出文件路径
dic = '/gyc_data/fray_data/Arbiter/tpch-userdef-kit/queries/10/'
input_file = '7_0.sql'   # 输入文件路径
output_file = 'output.txt' # 输出文件路径

# 打开输入文件，读取内容，并替换字符串
with open(dic + input_file, 'r') as file:
    content = file.read()

# 将修改后的内容写入输出文件
for i in range(50, 201):
    modified_content = content.replace('_1_prt_p0', f'_1_prt_p{i}')
    with open(dic + f"7_{i}.sql", 'w') as file:
        file.write(modified_content)
    print(f"替换完成结果已写入7_{i}.sql")