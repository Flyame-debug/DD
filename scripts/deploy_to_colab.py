# scripts/deploy_to_colab.py
import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse

class ColabDeployer:
    def __init__(self, local_project_root=None, colab_project_name=None):
        """
        初始化部署器
        
        Args:
            local_project_root: 本地项目根目录，默认为脚本所在目录的上两级
            colab_project_name: 在Colab中创建的项目目录名
        """
        # 1. 确定本地项目根目录
        if local_project_root is None:
            # 假设这个脚本在 scripts/ 目录下
            self.local_root = Path(__file__).resolve().parent.parent
        else:
            self.local_root = Path(local_project_root).resolve()
            
        # 2. 确定Colab中的项目目录
        self.colab_project_name = colab_project_name or self.local_root.name
        self.colab_root = Path('/content') / self.colab_project_name
        
        print(f"🚀 Colab 项目部署器初始化")
        print(f"   本地项目: {self.local_root}")
        print(f"   Colab目录: {self.colab_root}")
    
    def validate_local_project(self):
        """验证本地项目结构"""
        print("\n🔍 验证本地项目结构...")
        
        required_items = {
            'config/': '配置目录',
            'data/': '数据模块',
            'models/': '模型模块', 
            'training/': '训练模块',
            'main.py': '主入口文件',
            'requirements.txt': '依赖文件'
        }
        
        all_ok = True
        for item, desc in required_items.items():
            path = self.local_root / item.rstrip('/')
            if path.exists():
                print(f"   ✅ {desc}: {item}")
            else:
                print(f"   ❌ 缺失 {desc}: {item}")
                all_ok = False
        
        if not all_ok:
            print("\n⚠ 项目结构不完整，可能影响部署。")
        
        return all_ok
    
    def sync_to_colab(self, exclude_patterns=None):
        """
        将本地项目同步到Colab环境
        
        Args:
            exclude_patterns: 要排除的文件模式，如 ['__pycache__', '*.log', 'outputs/']
        """
        print(f"\n🔄 同步项目到 Colab...")
        
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '*.pyc', '.git', '.venv', 'outputs/', 'logs/']
        
        # 清理并创建Colab目录
        if self.colab_root.exists():
            shutil.rmtree(self.colab_root)
            print(f"   已清理旧目录: {self.colab_root}")
        
        self.colab_root.mkdir(parents=True, exist_ok=True)
        
        # 同步文件（排除指定模式）
        items_copied = 0
        for item in self.local_root.iterdir():
            item_name = item.name
            
            # 检查是否在排除列表中
            exclude = False
            for pattern in exclude_patterns:
                if pattern.endswith('/'):
                    if item_name == pattern.rstrip('/'):
                        exclude = True
                        break
                elif pattern.startswith('*'):
                    if item_name.endswith(pattern[1:]):
                        exclude = True
                        break
                elif item_name == pattern:
                    exclude = True
                    break
            
            if exclude:
                print(f"   ⏭ 跳过: {item_name}")
                continue
            
            dst = self.colab_root / item_name
            if item.is_dir():
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns(*exclude_patterns))
                print(f"   📁 复制目录: {item_name}")
                items_copied += 1
            elif item.is_file():
                shutil.copy2(item, dst)
                print(f"   📄 复制文件: {item_name}")
                items_copied += 1
        
        print(f"   ✅ 同步完成，共复制 {items_copied} 个项目")
        
        return self.colab_root
    
    def install_dependencies(self):
        """在Colab中安装项目依赖"""
        print(f"\n📦 安装Python依赖...")
        
        req_file = self.colab_root / 'requirements.txt'
        
        if req_file.exists():
            print(f"   从 requirements.txt 安装依赖...")
            result = subprocess.run(
                ['pip', 'install', '-r', str(req_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("   ✅ 依赖安装成功")
            else:
                print(f"   ⚠ 依赖安装可能有误: {result.stderr[:200]}")
        else:
            print(f"   ℹ 未找到 requirements.txt，安装基础依赖")
            # 安装基础深度学习依赖
            base_deps = [
                'torch',
                'torchvision', 
                'torchtext',
                'torchaudio',
                'efficientnet-pytorch',
                'albumentations',
                'pandas',
                'scikit-learn',
                'pyyaml',
                'tqdm'
            ]
            
            for dep in base_deps:
                subprocess.run(['pip', 'install', '-q', dep], 
                             capture_output=True)
                print(f"   ✅ 安装: {dep}")
    
    def setup_colab_environment(self):
        """设置Colab环境（挂载Drive、下载数据等）"""
        print(f"\n⚙ 设置Colab环境...")
        
        # 1. 挂载Google Drive（如果数据在Drive上）
        print(f"   1. 挂载Google Drive...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print(f"      ✅ Google Drive已挂载到 /content/drive")
            
            # 在项目目录创建指向Drive数据的软链接（可选）
            data_sources = [
                '/content/drive/MyDrive/chest_xray_data',
                '/content/drive/MyDrive/datasets/chest_xray',
                '/content/drive/MyDrive/data/chest_xray'
            ]
            
            for source in data_sources:
                if Path(source).exists():
                    target = self.colab_root / 'data' / 'chest_xray'
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(source, target)
                        print(f"      🔗 创建数据软链接: {source} -> {target}")
                    break
            
        except ImportError:
            print(f"      ℹ 不在Colab环境中，跳过Drive挂载")
        except Exception as e:
            print(f"      ⚠ Drive挂载失败: {e}")
        
        # 2. 设置Python路径
        print(f"   2. 设置Python路径...")
        sys.path.insert(0, str(self.colab_root))
        os.chdir(self.colab_root)
        print(f"      ✅ 工作目录: {os.getcwd()}")
        print(f"      ✅ Python路径已添加: {self.colab_root}")
    
    def run_training(self, training_args=None):
        """在Colab中运行训练"""
        print(f"\n🚀 开始训练...")
        
        if training_args is None:
            training_args = ['--mode', 'train', '--colab']
        
        # 切换到项目目录
        os.chdir(self.colab_root)
        
        # 构建命令
        cmd = ['python', 'main.py'] + training_args
        cmd_str = ' '.join(cmd)
        print(f"   执行命令: {cmd_str}")
        
        # 执行训练
        print(f"   {'='*50}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(f"   {line}", end='')
        
        process.wait()
        print(f"   {'='*50}")
        
        if process.returncode == 0:
            print(f"   ✅ 训练完成!")
        else:
            print(f"   ❌ 训练失败，退出码: {process.returncode}")
        
        return process.returncode
    
    def download_results(self, local_output_dir=None):
        """将训练结果下载回本地（需要在Colab中运行）"""
        print(f"\n📥 下载训练结果...")
        
        try:
            from google.colab import files
            
            # 压缩输出目录
            outputs_dir = self.colab_root / 'outputs'
            if outputs_dir.exists():
                import tarfile
                
                # 创建压缩包
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"training_results_{timestamp}.tar.gz"
                archive_path = self.colab_root / archive_name
                
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(outputs_dir, arcname='outputs')
                
                print(f"   已创建结果压缩包: {archive_path}")
                
                # 下载
                files.download(str(archive_path))
                print(f"   ✅ 已开始下载")
            else:
                print(f"   ℹ 未找到输出目录: {outputs_dir}")
                
        except ImportError:
            print(f"   ℹ 不在Colab环境中，无法下载")
        except Exception as e:
            print(f"   ⚠ 下载失败: {e}")

def main():
    """主函数：一键部署并训练"""
    parser = argparse.ArgumentParser(description='部署项目到Colab并训练')
    parser.add_argument('--local-dir', help='本地项目目录路径')
    parser.add_argument('--colab-dir', help='Colab中的项目目录名')
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                       help='运行模式')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 自动部署到Colab训练系统")
    print("="*60)
    
    # 创建部署器
    deployer = ColabDeployer(
        local_project_root=args.local_dir,
        colab_project_name=args.colab_dir
    )
    
    # 验证项目
    if not deployer.validate_local_project():
        response = input("项目结构不完整，继续部署吗？(y/n): ")
        if response.lower() != 'y':
            return
    
    # 同步项目到Colab
    deployer.sync_to_colab()
    
    # 安装依赖
    deployer.install_dependencies()
    
    # 设置环境
    deployer.setup_colab_environment()
    
    # 构建训练参数
    training_args = ['--mode', args.mode, '--colab']
    if args.epochs:
        training_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        training_args.extend(['--batch-size', str(args.batch_size)])
    
    # 运行训练
    deployer.run_training(training_args)
    
    # 提示下载结果
    print("\n" + "="*60)
    print("📋 训练完成!")
    print("="*60)
    print("下一步操作：")
    print("1. 结果文件保存在Colab的 outputs/ 目录")
    print("2. 如需下载到本地，运行: deployer.download_results()")
    print("3. 或手动从Colab文件浏览器下载")

if __name__ == '__main__':
    from datetime import datetime
    main()