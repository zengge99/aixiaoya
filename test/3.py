# 给单文件提取增加1次重试
def extract_single_file_to_memory(archive, filename):
    try:
        bio = io.BytesIO()
        archive.extract(targets=[filename], path=bio)
        bio.seek(0)
        file_path = os.path.join(bio.name, filename)
        with open(file_path, 'rb') as f:
            data = f.read()
        os.remove(file_path)
        bio.close()
        return data
    except Exception as e:
        # 重试1次
        try:
            bio = io.BytesIO()
            archive.extract(targets=[filename], path=bio)
            bio.seek(0)
            file_path = os.path.join(bio.name, filename)
            with open(file_path, 'rb') as f:
                data = f.read()
            os.remove(file_path)
            bio.close()
            return data
        except:
            return None