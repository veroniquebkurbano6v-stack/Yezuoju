# verify_key.py
import os
import wandb

print("检查W&B API Key配置...")
print("=" * 50)

# 方法1：环境变量
api_key = os.environ.get('WANDB_API_KEY')
if api_key:
    print(f"✅ 环境变量WANDB_API_KEY存在")
    print(f"   长度: {len(api_key)} 字符")
    print(f"   内容: {api_key[:10]}...{api_key[-10:]}")
else:
    print("❌ 环境变量WANDB_API_KEY不存在")

# 方法2：netrc文件
import netrc

try:
    nrc = netrc.netrc()
    machines = nrc.hosts
    if 'api.wandb.ai' in machines:
        print("✅ netrc文件中找到api.wandb.ai配置")
        login, account, password = nrc.authenticators('api.wandb.ai')
        print(f"   API Key长度: {len(password)} 字符")
    else:
        print("❌ netrc文件中未找到api.wandb.ai配置")
except Exception as e:
    print(f"❌ 读取netrc文件失败: {e}")

# 方法3：测试登录
print("\n测试W&B登录...")
try:
    # 设置临时环境变量（如果还没有）
    if not api_key:
        test_key = "test_key"  # 假key用于测试
        os.environ['WANDB_API_KEY'] = test_key

    wandb.login()
    print("✅ W&B登录成功")
except Exception as e:
    print(f"❌ W&B登录失败: {e}")

print("=" * 50)
print("建议的正确API Key格式:")
print("长度: 40个字符")
print("示例: bf55badf1a1c1b8f0eb52313bd403fdc90eba61e")
