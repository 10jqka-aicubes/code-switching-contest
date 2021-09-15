#!/usr/bin/env bash

# Copyright 2021 Hithink RoyalFlush Information Network Co.,Ltd

. ./cmd.sh

stage=-1

aishell_data=/cpu4/16k_ch/Aishell-170h/data/
librispeech_data=/cpu4/16k_en/Librispeech/data/

if [ $stage -le -1 ]; then

  echo "This recipe uses mixed phoneme strategy. "
  echo "stage -1: data preparation."
  local/aishell_data_prep.sh ${aishell_data}/wav ${aishell_data}/transcript
  local/librispeech_data_prep.sh $librispeech_data/LibriSpeech/train-clean-100 data/$(echo "train-clean-100" | sed s/-/_/g)
fi

if [ ${stage} -le 0 ] ; then

    # Data prepare lang.
    echo "stage 0: data preparation."
    # when the "--stage 3" option is used below we skip the G2P steps, and use the
    # lexicon we have already downloaded from openslr.org/11/
    local/librispeech_prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
                                      data/local/lm data/local/lm data/local/dict_nosp
    local/aishell_prepare_dict.sh $aishell_data/resource_aishell || exit 1;
fi

if [ ${stage} -le 1 ] ; then
   # combine the phones sets
   mkdir -p data/dict_ench
   cat data/local/dict_nosp/extra_questions.txt data/local/dict/extra_questions.txt | sed  '/SIL/d' > data/dict_ench/extra_questions.txt
   cat data/local/dict_nosp/lexicon.txt data/local/dict/lexicon.txt  > data/dict_ench/lexicon.txt
   cat data/local/dict_nosp/nonsilence_phones.txt data/local/dict/nonsilence_phones.txt  > data/dict_ench/nonsilence_phones.txt
   cat data/local/dict_nosp/optional_silence.txt data/local/dict/optional_silence.txt | sed  '/SIL/d'  > data/dict_ench/optional_silence.txt
   cat data/local/dict_nosp/silence_phones.txt data/local/dict/silence_phones.txt  > data/dict_ench/silence_phones.txt
   #sed -i '/SIL/d' data/dict_ench/extra_questions.txt
fi

if [ ${stage} -le 2 ] ; then
  utils/prepare_lang.sh --position_dependent_phones false data/dict_ench "<UNK>" data/local/lang data/lang || exit 1;
fi

if [ ${stage} -le 3 ] ; then

  # format train dir
  ./utils/data/combine_data.sh data/Aishell data/train data/dev
  mv data/train_clean_100 data/LibriClean
  mkdir -p data/tmp
  mv data/test data/Aishell_test
  mv data/train data/dev data/tmp
fi

if [ $stage -le 4 ]; then
  mfccdir=mfcc
  for x in Aishell  LibriClean; do
    utils/fix_data_dir.sh data/$x || exit 1;
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/$x exp/make_mfcc/$x exp/$mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x exp/$mfccdir || exit 1;
  done
fi

if [ $stage -le 5 ]; then
  # Extract data
  utils/subset_data_dir.sh --shortest data/Aishell 3000 data/Aishell_3k || exit 1;
  utils/subset_data_dir.sh --shortest data/LibriClean 2000 data/LibriClean_2k || exit 1;
  # combine
  utils/data/combine_data.sh data/mix_5k data/Aishell_3k data/LibriClean_2k
fi

if [ $stage -le 6 ]; then
  # Monophone training
  steps/train_mono.sh --boost-silence 1.25 --cmd "$train_cmd" --nj 20 \
    data/mix_5k data/lang exp/mono || exit 1;
  # Monophone decoding
  # utils/mkgraph.sh data/graph/lang_ench exp/mono exp/mono/graph_ench || exit 1
  # for item in dev test ; do
  #   steps/decode.sh --cmd "$decode_cmd" --nj 20 exp/mono/graph_ench data/${item} exp/mono/decode_${item}
  # done
fi

if [ $stage -le 7 ]; then
  # Extract data
  utils/subset_data_dir.sh data/Aishell 15000 data/Aishell_15k || exit 1;
  utils/subset_data_dir.sh data/LibriClean 5000 data/LibriClean_5k || exit 1;
  # Combine
  utils/data/combine_data.sh data/mix_20k data/Aishell_15k data/LibriClean_5k
fi

if [ $stage -le 8 ]; then
  # Get alignments from monophone system.
  steps/align_si.sh --boost-silence 1.25 --cmd "$train_cmd" --nj 10 \
                    data/mix_20k data/lang exp/mono exp/mono_ali || exit 1;
  # Train tri1
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/mix_20k data/lang exp/mono_ali exp/tri1 || exit 1;
  # Decode tri1
  # utils/mkgraph.sh data/graph/lang_ench exp/tri1 exp/tri1/graph_ench || exit 1;
  # for item in dev test ; do
  #   steps/decode.sh --cmd "$decode_cmd" --nj 20 exp/tri1/graph_ench data/${item} exp/tri1/decode_${item}
  # done
fi

if [ $stage -le 9 ]; then
  # Extract data
  utils/subset_data_dir.sh data/Aishell 40000 data/Aishell_40k || exit 1;
  utils/subset_data_dir.sh data/LibriClean 10000 data/LibriClean_10k || exit 1;
  # Combine
  utils/data/combine_data.sh data/mix_50k data/Aishell_40k data/LibriClean_10k
fi

if [ $stage -le 10 ]; then
  # Get aliments from tri1
  steps/align_si.sh --cmd "$train_cmd" --nj 20 \
                    data/mix_50k data/lang exp/tri1 exp/tri1_ali_50k || exit 1;
  # Train tri2, which is LDA+MLLT,
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" \
                          2500 15000 data/mix_50k data/lang exp/tri1_ali_50k exp/tri2 || exit 1;
  # Decode tri2
  # utils/mkgraph.sh data/graph/lang_ench exp/tri2 exp/tri2/graph_ench || exit 1;
  # for item in dev test ; do
  #   steps/decode.sh --cmd "$decode_cmd" --nj 20 exp/tri2/graph_ench data/${item} exp/tri2/decode_${item}
  # done
fi

if [ $stage -le 11 ]; then
  # Extract data
  utils/subset_data_dir.sh data/Aishell 80000 data/Aishell_80k || exit 1;
  utils/subset_data_dir.sh data/LibriClean 20000 data/LibriClean_20k || exit 1;
  # Combine
  utils/data/combine_data.sh data/mix_100k data/Aishell_80k data/LibriClean_20k
fi

if [ $stage -le 12 ]; then
  # Get aliments from tri2
  steps/align_si.sh --cmd "$train_cmd" --nj 20 \
                    data/mix_100k data/lang exp/tri2 exp/tri2_ali || exit 1;
  # Train tri3
  steps/train_sat.sh --cmd "$train_cmd" \
                     2500 15000 data/mix_100k data/lang exp/tri2_ali exp/tri3 || exit 1;
  # Decode tri3
  # utils/mkgraph.sh data/graph/lang_ench exp/tri3 exp/tri3/graph_ench || exit 1;
  # for item in dev test ; do
  #   steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 20 exp/tri3/graph_ench data/${item} exp/tri3/decode_${item}
  # done
fi

if [ $stage -le 13 ]; then
  # Combine
  utils/data/combine_data.sh data/mix_all data/Aishell data/LibriClean
fi

if [ $stage -le 14 ]; then
  # Ali
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 20 \
                       data/mix_all data/lang exp/tri3 exp/tri3_ali || exit 1;
  # Train
  steps/train_sat.sh --cmd "$train_cmd" \
                     4000 40000 data/mix_all data/lang exp/tri3_ali exp/tri4 || exit 1;
  # Decode tri4
  # utils/mkgraph.sh data/graph/lang_ench exp/tri4 exp/tri4/graph_ench || exit 1;
  # for item in dev test ; do
  #   steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 20 exp/tri4/graph_ench data/${item} exp/tri4/decode_${item}
  # done
fi

if [ $stage -le 15 ]; then
  
  # Speed perturb
  utils/data/perturb_data_dir_speed_3way.sh data/mix_all data/mix_all_sp
  # mix_all_sp for mfcc, train_all_sp for fbank
  cp -r data/mix_all_sp data/train_all_sp

  # Generate feats
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/mix_all_sp exp/make_mfcc/mix_all_sp exp/mfcc || exit 1;
  steps/compute_cmvn_stats.sh data/mix_all_sp exp/make_mfcc/mix_all_sp exp/mfcc || exit 1;

  steps/make_fbank.sh --nj 20 --cmd "$train_cmd" data/train_all_sp exp/make_fbank/train_all_sp exp/fbank
  steps/compute_cmvn_stats.sh data/train_all_sp exp/make_fbank/train_all_sp exp/fbank
  # Ali
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 20 \
                       data/mix_all_sp data/lang exp/tri4 exp/tri4_ali || exit 1;
fi

# baseline chain model
local/chain/run_tdnn_baseline.sh

exit 0;
