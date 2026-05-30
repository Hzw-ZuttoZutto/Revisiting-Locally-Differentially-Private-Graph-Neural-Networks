# bash ./suite_main_add.sh      # 跑完主实验额外的隐私预算
/home/hzw/miniconda3/envs/HZWDP/bin/python scripts/backfill_main_add_10seed.py --worker-ids 0,1,2,3,4,5 #完整跑全两外5个随机种子
bash ./suite_main2_again.sh   # 跑完两个feature-free baseline，分别用于中间的striking finding, 以及后续的topoloy mainly 的理论说明
bash ./suite_figure4_again.sh # 跑完random code replacements，用于topology mainly 的理论说明
bash ./suite_figure5_again.sh # 跑完连续纠偏消融，用于rectification 的理论说明
bash ./suite_figure0.sh # 完成clean reference