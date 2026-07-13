import os
from pathlib import Path

# 设置基础路径
base_dir = Path("/gpfsdata/home/renyixiang/SkinDB/10X/human")
ref_dir = "/gpfsdata/home/renyixiang/cellranger/refdata-gex-GRCh38-2020-A"
sample_prefix = "mysample"
overwrite = False  # 是否重写已有脚本

def create_cellranger_script(parent_dir, sample_dir, sample_id, parent_id, base_dir, ref_dir, sample_prefix):
    """
    创建 Cell Ranger 脚本的通用函数
    
    Args:
        parent_dir: 父目录路径 (GSE/HRA)
        sample_dir: 样本目录路径 (GSM/HRR)
        sample_id: 样本ID (GSM***/HRR***)
        parent_id: 父目录ID (GSE***/HRA***)
        base_dir: 基础目录
        ref_dir: 参考基因组目录
        sample_prefix: FASTQ 样本前缀
    """
    script_path = sample_dir / f"run_cellranger_{sample_id}.sh"
    
    if script_path.exists() and not overwrite:
        print(f"✅ 已存在脚本: {script_path.name}，跳过")
        return
    
    # 构建脚本内容
    script_content = f"""#!/bin/bash
# === 自动生成的 Cell Ranger 脚本 ===
SAMPLE_ID="{sample_id}"
FASTQ_SAMPLE_NAME="{sample_prefix}"
PARENT_ID="{parent_id}"
BASE_DIR="{base_dir}"
REF_DIR="{ref_dir}"

PARENT_DIR="${{BASE_DIR}}/${{PARENT_ID}}"
SAMPLE_DIR="${{PARENT_DIR}}/${{SAMPLE_ID}}"
RUN_ID="run_count_${{SAMPLE_ID}}"
LOG_FILE="${{PARENT_DIR}}/run_cellranger_${{SAMPLE_ID}}.log"

cd "${{PARENT_DIR}}"

echo "[INFO] Starting Cell Ranger for ${{SAMPLE_ID}} at $(date)"

nohup cellranger count \\
  --id="${{RUN_ID}}" \\
  --fastqs="${{SAMPLE_DIR}}" \\
  --transcriptome="${{REF_DIR}}" \\
  --sample="${{FASTQ_SAMPLE_NAME}}" \\
  --no-bam \\
  --localcores=4 \\
  --localmem=96 \\
  --noexit > "${{LOG_FILE}}" 2>&1 &

echo "[INFO] Cell Ranger started in background. Log: ${{LOG_FILE}}"
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ {'重写' if script_path.exists() and overwrite else '已创建'}脚本: {script_path}")

# 遍历 GSE/GSM 文件夹
print("\n=== 处理 GSE/GSM 文件夹 ===")
for gse_dir in base_dir.iterdir():
    if not gse_dir.is_dir() or not gse_dir.name.startswith("GSE"):
        continue
    
    # 遍历 GSM 子文件夹
    for gsm_dir in gse_dir.iterdir():
        if not gsm_dir.is_dir() or not gsm_dir.name.startswith("GSM"):
            continue
        
        create_cellranger_script(
            parent_dir=gse_dir,
            sample_dir=gsm_dir,
            sample_id=gsm_dir.name,
            parent_id=gse_dir.name,
            base_dir=base_dir,
            ref_dir=ref_dir,
            sample_prefix=sample_prefix
        )

# 遍历 HRA/HRR 文件夹
print("\n=== 处理 HRA/HRR 文件夹 ===")
for hra_dir in base_dir.iterdir():
    if not hra_dir.is_dir() or not hra_dir.name.startswith("HRA"):
        continue
    
    # 遍历 HRR 子文件夹
    for hrr_dir in hra_dir.iterdir():
        if not hrr_dir.is_dir() or not hrr_dir.name.startswith("HRR"):
            continue
        
        create_cellranger_script(
            parent_dir=hra_dir,
            sample_dir=hrr_dir,
            sample_id=hrr_dir.name,
            parent_id=hra_dir.name,
            base_dir=base_dir,
            ref_dir=ref_dir,
            sample_prefix=sample_prefix
        )

print("\n✅ 所有脚本处理完成！")