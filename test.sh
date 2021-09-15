#!/usr/bin/env bash

# Copyright 2021 Hithink RoyalFlush Information Network Co.,Ltd

. ./cmd.sh


# dev data config
dev_data_dir=data/corpus/THS-DEV
# model config
dir=exp/chain/tdnn_affix_baseline

# prepare dev data
python3 local/generate_kaldi_data_prep_from_wav_txt.py $dev_data_dir data/THS_DEV

# Extract feats
steps/make_fbank.sh --nj 20 --cmd "$train_cmd" data/THS_DEV exp/make_fbank/THS_DEV exp/fbank
steps/compute_cmvn_stats.sh data/THS_DEV exp/make_fbank/THS_DEV exp/fbank

# make graph
graph_dir=$dir/graph_ench
utils/format_lm.sh data/lang data/LM/v1.lm.gz data/dict_ench/lexicon.txt ${graph_dir} || exit 1;
utils/mkgraph.sh  --self-loop-scale 1.0 data/graph/lang_ench  $dir ${graph_dir}

# decode
data_test="THS_DEV"

steps/nnet3/decode.sh --stage 0 --nj 10 --cmd "$decode_cmd" \
                      --acwt 1.0 --post-decode-acwt 10.0 \
                      $dir/graph_ench ${data_test} $dir/decode_${data_test} || exit 1;


for x in $dir/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

python3 compute-mer.py $dir/decode_test/scoring_kaldi

echo "THS-DEV decode done"
