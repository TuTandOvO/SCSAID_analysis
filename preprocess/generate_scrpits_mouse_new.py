# -*- coding: utf-8 -*-
import os
from pathlib import Path

# 设置基础路径
base_dir = Path("/gpfsdata/home/renyixiang/SkinDB/10X/mouse")
ref_dir = "/gpfsdata/home/renyixiang/cellranger/refdata-gex-mm10-2020-A"
sample_prefix = "mysample"
overwrite = False  # 是否重写已有脚本

# ===== 遍历 GSE/**/GSM/** =====
for gse_dir in base_dir.iterdir():
    if not gse_dir.is_dir() or not gse_dir.name.startswith("GSE"):
        continue

    # 遍历 GSM 子文件夹
    for gsm_dir in gse_dir.iterdir():
        if not gsm_dir.is_dir() or not gsm_dir.name.startswith("GSM"):
            continue

        gsm_id = gsm_dir.name
        gse_id = gse_dir.name
        script_path = gsm_dir / f"run_cellranger_{gsm_id}.sh"

        if script_path.exists() and not overwrite:
            print(f"✅ 已存在脚本: {script_path.name}，跳过")
            continue

        # 构建脚本内容（已修复 --id 参数）
        script_content = f"""#!/bin/bash

# === 自动生成的 Cell Ranger 脚本 ===
GSM_ID="{gsm_id}"
FASTQ_SAMPLE_NAME="{sample_prefix}"
GSE_ID="{gse_id}"
BASE_DIR="{base_dir}"
REF_DIR="{ref_dir}"

GSE_DIR="${{BASE_DIR}}/${{GSE_ID}}"
GSM_DIR="${{GSE_DIR}}/${{GSM_ID}}"
RUN_ID="run_count_${{GSM_ID}}"
LOG_FILE="${{GSE_DIR}}/run_cellranger_${{GSM_ID}}.log"

cd "${{GSE_DIR}}"

echo "[INFO] Starting Cell Ranger for ${{GSM_ID}} at $(date)"
nohup cellranger count \\
  --id="${{RUN_ID}}" \\
  --fastqs="${{GSM_DIR}}" \\
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

# ===== 追加遍历 CRA/**/SAMC/** =====
for cra_dir in base_dir.iterdir():
    if not cra_dir.is_dir() or not cra_dir.name.startswith("CRA"):
        continue

    # 遍历 SAMC 子文件夹
    for samc_dir in cra_dir.iterdir():
        if not samc_dir.is_dir() or not samc_dir.name.startswith("SAMC"):
            continue

        samc_id = samc_dir.name
        cra_id = cra_dir.name
        script_path = samc_dir / f"run_cellranger_{samc_id}.sh"

        if script_path.exists() and not overwrite:
            print(f"✅ 已存在脚本: {script_path.name}，跳过")
            continue

        # 构建脚本内容（CRA/SAMC 版本）
        script_content = f"""#!/bin/bash

# === 自动生成的 Cell Ranger 脚本 ===
SAMC_ID="{samc_id}"
FASTQ_SAMPLE_NAME="{sample_prefix}"
CRA_ID="{cra_id}"
BASE_DIR="{base_dir}"
REF_DIR="{ref_dir}"

CRA_DIR="${{BASE_DIR}}/${{CRA_ID}}"
SAMC_DIR="${{CRA_DIR}}/${{SAMC_ID}}"
RUN_ID="run_count_${{SAMC_ID}}"
LOG_FILE="${{CRA_DIR}}/run_cellranger_${{SAMC_ID}}.log"

cd "${{CRA_DIR}}"

echo "[INFO] Starting Cell Ranger for ${{SAMC_ID}} at $(date)"
nohup cellranger count \\
  --id="${{RUN_ID}}" \\
  --fastqs="${{SAMC_DIR}}" \\
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
