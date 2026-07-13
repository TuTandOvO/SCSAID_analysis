#!/usr/bin/env bash
set -euo pipefail

TSV_FILE="./new_human_by_srr.tsv"
OUTDIR="$(pwd)"
THREADS=20  # 增加线程数以更好利用CPU
COMPRESS_FASTQ=true
MIN_BYTES=1024
LOG_FILE="./download.log"
MAX_SIZE="500G"  # 增加最大下载大小限制

trap 'echo "[INTERRUPT] stopping..."; pkill -9 -f fasterq-dump || true; exit 130' INT

# 初始化日志
echo "===== $(date) START DOWNLOAD =====" >> "$LOG_FILE"

# 从表头提取 GSE/GSM/SRR 三列
awk -F'\t' '
NR==1{
  for(i=1;i<=NF;i++){
    h=toupper($i)
    if(!cG && (h=="GSE" || h ~ /SERIES/)) cG=i
    if(!cM && (h=="GSM" || h ~ /SAMPLE/)) cM=i
    if(!cR && (h=="SRR" || h=="RUN" || h=="RUN_ACCESSION")) cR=i
  }
  if(!cG || !cM || !cR){ print "ERROR: 找不到 GSE/GSM/SRR 列" > "/dev/stderr"; exit 1 }
  next
}
{
  g=$cG; m=$cM; r=$cR
  if (match(r, /(SRR[0-9]+)/, a)) r=a[1]; else next
  print g "\t" m "\t" r
}' "$TSV_FILE" > .tmp_triplets.tsv
sed -i 's/\r$//' .tmp_triplets.tsv

# 检查必须有 _1 和 _2
have_fastqs() {
  local dir="$1"; local srr="$2"
  for ext in fastq fastq.gz; do
    f1="${dir}/${srr}_1.${ext}"
    f2="${dir}/${srr}_2.${ext}"
    if [[ -s "$f1" && $(stat -c%s "$f1") -ge $MIN_BYTES \
       && -s "$f2" && $(stat -c%s "$f2") -ge $MIN_BYTES ]]; then
      return 0
    fi
  done
  return 1
}

while IFS=$'\t' read -r GSE GSM SRR; do
  [[ -z "${SRR:-}" ]] && continue
  gsm_dir="${OUTDIR}/${GSE}/${GSM}"
  mkdir -p "$gsm_dir"

  echo "[INFO] Processing ${SRR} -> ${gsm_dir}"

  # 已有 _1/_2 就跳过
  if have_fastqs "$gsm_dir" "$SRR"; then
    echo "[SKIP] ${SRR} 已存在 _1/_2 fastq"
    echo -e "$(date)\t[SKIP]\t${SRR}\t${GSE}\t${GSM}" >> "$LOG_FILE"
    # 补压缩
    if [[ "${COMPRESS_FASTQ}" == true ]]; then
      if command -v pigz >/dev/null 2>&1; then
        pigz -p "$THREADS" -f "${gsm_dir}/${SRR}"_*.fastq 2>/dev/null || true
      else
        gzip -f "${gsm_dir}/${SRR}"_*.fastq 2>/dev/null || true
      fi
    fi
    continue
  fi

  # 检查是否已有本地 SRA 文件
  sra_here=""
  if [[ -s "${gsm_dir}/${SRR}.sra" ]]; then
    sra_here="${gsm_dir}/${SRR}.sra"
    echo "[INFO] 使用本地 SRA 文件: $sra_here"
  fi

  # 尝试直接流式处理，不下载 SRA 文件
  echo "[INFO] 尝试流式处理 ${SRR}..."
  if [[ -n "$sra_here" ]]; then
    # 使用本地文件，增加线程数和临时目录参数
    dump_cmd="fasterq-dump \"$sra_here\" --split-files --include-technical -e $THREADS -p -O \"$gsm_dir\" --temp \"$gsm_dir\""
  else
    # 直接从远程流式处理，增加线程数和临时目录参数
    dump_cmd="fasterq-dump \"$SRR\" --split-files --include-technical -e $THREADS -p -O \"$gsm_dir\" --temp \"$gsm_dir\""
  fi

  if eval "$dump_cmd"; then
    echo "[INFO] 流式处理成功: $SRR"
  else
    echo "[WARN] 流式处理失败，尝试先下载 SRA 文件..."
    
    # 如果流式处理失败，尝试先下载
    if [[ -z "$sra_here" ]]; then
      dl_tmp="${gsm_dir}/.sra_dl"; rm -rf "$dl_tmp"; mkdir -p "$dl_tmp"
      # 使用更大的size限制和增加重试机制
      if prefetch "$SRR" --output-directory "$dl_tmp" --max-size "$MAX_SIZE" --force all; then
        sra_path="$(find "$dl_tmp" -type f -name "${SRR}.sra" -print -quit || true)"
        [[ -z "$sra_path" ]] && sra_path="$(find "$dl_tmp" -type f -name '*.sra' -print -quit || true)"
        if [[ -n "$sra_path" ]]; then
          mv -f "$sra_path" "${gsm_dir}/${SRR}.sra"
          sra_here="${gsm_dir}/${SRR}.sra"
          rm -rf "$dl_tmp"
          
          # 再次尝试处理，使用更多线程和临时目录
          if ! fasterq-dump "$sra_here" --split-files --include-technical -e "$THREADS" -p -O "$gsm_dir" --temp "$gsm_dir"; then
            echo "[WARN] fasterq-dump 失败：$SRR"
            echo -e "$(date)\t[FAIL]\t${SRR}\t${GSE}\t${GSM}\tfasterq-dump失败" >> "$LOG_FILE"
            continue
          fi
        else
          echo "[WARN] 未找到 SRA 文件：$SRR"
          echo -e "$(date)\t[FAIL]\t${SRR}\t${GSE}\t${GSM}\t未找到SRA" >> "$LOG_FILE"
          rm -rf "$dl_tmp"
          continue
        fi
      else
        echo "[WARN] prefetch 失败：$SRR"
        echo -e "$(date)\t[FAIL]\t${SRR}\t${GSE}\t${GSM}\tprefetch失败" >> "$LOG_FILE"
        rm -rf "$dl_tmp"
        continue
      fi
    else
      echo "[WARN] fasterq-dump 失败：$SRR"
      echo -e "$(date)\t[FAIL]\t${SRR}\t${GSE}\t${GSM}\tfasterq-dump失败" >> "$LOG_FILE"
      continue
    fi
  fi

  # 使用更多线程进行压缩
  if [[ "${COMPRESS_FASTQ}" == true ]]; then
    if command -v pigz >/dev/null 2>&1; then
      pigz -p "$THREADS" -f "${gsm_dir}/${SRR}"_*.fastq 2>/dev/null || true
    else
      gzip -f "${gsm_dir}/${SRR}"_*.fastq 2>/dev/null || true
    fi
  fi

  if have_fastqs "$gsm_dir" "$SRR"; then
    echo "[OK] ${SRR} done."
    echo -e "$(date)\t[OK]\t${SRR}\t${GSE}\t${GSM}" >> "$LOG_FILE"
    # 处理完成后可以选择删除 SRA 文件以节省空间
    # rm -f "${gsm_dir}/${SRR}.sra"
  else
    echo "[WARN] ${SRR} fastq 不完整"
    echo -e "$(date)\t[FAIL]\t${SRR}\t${GSE}\t${GSM}\tfastq不完整" >> "$LOG_FILE"
  fi
done < .tmp_triplets.tsv

echo "===== $(date) ALL DONE =====" >> "$LOG_FILE"
echo "[ALL DONE]"
