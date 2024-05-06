import os

def delete_dot_files(directory):
    """
    删除指定目录及其子目录中所有以点(.)开头的文件。
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        # 遍历所有文件
        for name in files:
            if name.startswith('.'):
                file_path = os.path.join(root, name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        # 可选：如果你也想删除以点(.)开头的文件夹，取消下面代码的注释
        # for name in dirs:
        #     if name.startswith('.'):
        #         dir_path = os.path.join(root, name)
        #         os.rmdir(dir_path)  # 注意，只能删除空文件夹
        #         print(f"Deleted directory: {dir_path}")

# 使用示例
# 替换 'your_directory_path' 为你想要清理的目录路径
delete_dot_files('G:/IEMOCAP')
