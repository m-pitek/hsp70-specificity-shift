#!/bin/bash

./fastp -i ./rawdata/IL-4_S1_L007_R1_001.fastq.gz -I ./rawdata/IL-4_S1_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-4_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-4_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-4_fastp.json --html ./fastp_filtered/IL-4_fastp.html

./fastp -i ./rawdata/IL-5_S2_L007_R1_001.fastq.gz -I ./rawdata/IL-5_S2_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-5_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-5_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-5_fastp.json --html ./fastp_filtered/IL-5_fastp.html

./fastp -i ./rawdata/IL-6_S3_L007_R1_001.fastq.gz -I ./rawdata/IL-6_S3_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-6_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-6_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-6_fastp.json --html ./fastp_filtered/IL-6_fastp.html

./fastp -i ./rawdata/IL-7_S4_L007_R1_001.fastq.gz -I ./rawdata/IL-7_S4_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-7_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-7_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-7_fastp.json --html ./fastp_filtered/IL-7_fastp.html

./fastp -i ./rawdata/IL-8_S5_L007_R1_001.fastq.gz -I ./rawdata/IL-8_S5_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-8_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-8_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-8_fastp.json --html ./fastp_filtered/IL-8_fastp.html

./fastp -i ./rawdata/IL-9_S6_L007_R1_001.fastq.gz -I ./rawdata/IL-9_S6_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-9_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-9_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-9_fastp.json --html ./fastp_filtered/IL-9_fastp.html

./fastp -i ./rawdata/IL-10_S7_L007_R1_001.fastq.gz -I ./rawdata/IL-10_S7_L007_R2_001.fastq.gz -o ./fastp_filtered/IL-10_Q25_filtered_R1.fq.gz -O ./fastp_filtered/IL-10_Q25_filtered_R2.fq.gz -q 25 -u 10 --disable_adapter_trimming  --dont_eval_duplication --json ./fastp_filtered/IL-10_fastp.json --html ./fastp_filtered/IL-10_fastp.html

