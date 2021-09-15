# 【9-10双月赛】基于单语种语料的中英混合语音识别算法



## 任务描述
利用开源的纯中文数据集Aishell-1 和纯英文数据Librispeech train_clean_100h 训练中英文混合的语音识别模型，解码包含中英文混合句子的测试集；
（可以利用提供的验证集进行模型自适应训练）

## 数据 
数据量：LibriSpeech train_clean_100h（英文） Aishell-1中训练集 （中文），提供的验证集语料

数据来源：www.openslr.org/33  www.openslr.org/12

标签说明：语音对应文本内容

字段说明：

## 算法

## 项目结构
```bash
CodeSwitchCompetition_Baseline_kaldi        # 项目名称 
|- conf/        	                        # 脚本参数
|-data
  |-LM                                      # 语言模型
|- local                                    # 
|- steps                                    # 通用功能脚本
|- utils                                    # 通用处理脚本
|- README.md                                # 项目说明
|- requirements.txt                         # 需要额外安装的python库, 要求使用
|
```

## 运行
```bash
# 训练&验证
cd code-switching_contest && ./run.sh

# 评测
cd code-switching_contest && ./test.sh
```

## 交付结果
开发环境要求: python3 & python 2.7

1. 源码


2. 预测结果：

   输出文件名: best_wer<br>
     &nbsp; 文件格式说明： <br>
     &nbsp; %WER 中文字错误率<br>

   输出文件名: best_mer<br>
     &nbsp; 文件格式说明：<br>
     &nbsp; CH: %CER 中文字错误率<br>
     &nbsp; EN: %WER 英文单词错误率<br>
     &nbsp; MIX: %MER 混合错误率<br>

3. 完成report.md 描述项目的优化思路


## 评估标准

离线评测指标为整体混合错误率（MER）：<br>
中文以汉字作为统计单元，计算字错误率（Character Error Rate， CER）；<br>
英文以单词作为统计单元，计算词错误率(Word Error Rate, WER 不对标点等符号进行统计）<br>

MER=(E_1+E_2)/(N_1+N_2 )<br>
CER=(Ins_1+ Del_1+ Sub_1)/N_1<br> 
WER=  (Ins_2+Del_2+Sub_2)/N_2 <br>


## 基线
当前模型值:
base: MER:72.95% CER:70.52% WER:72.95<br>
smbr: MER:38.71% CER:29.27% WER:75.76%<br>

预期目标值:
MER<30%
