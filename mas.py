import os
from api.main import mas

if __name__ == '__main__':
    # 设置环境变量来抑制HuggingFace的警告
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 运行主程序
    mas()