#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3 with xconfigs.


# local/nnet3/compare_wer.sh exp/nnet3/tdnn1a_sp
# System                tdnn1a_sp
#WER dev93 (tgpr)                9.18
#WER dev93 (tg)                  8.59
#WER dev93 (big-dict,tgpr)       6.45
#WER dev93 (big-dict,fg)         5.83
#WER eval92 (tgpr)               6.15
#WER eval92 (tg)                 5.55
#WER eval92 (big-dict,tgpr)      3.58
#WER eval92 (big-dict,fg)        2.98
# Final train prob        -0.7200
# Final valid prob        -0.8834
# Final train acc          0.7762
# Final valid acc          0.7301

set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=13
nj=20

train_set=train_si284
test_sets="test_dev93 test_eval92"
        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.
tdnn_affix=1a  #affix for TDNN directory e.g. "1a" or "1b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
remove_egs=false
srand=0
reporting_email=
# set common_egs_dir to use previously dumped egs.
common_egs_dir=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm=tri5b
gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali
dir=exp/nnet3/tdnn_nnet3_train_16K-fbank-pitch
train_data_dir=data/train_all_16K_fbank_pitch


if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=650
  relu-renorm-layer name=tdnn2 dim=650 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=650 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=650 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=650 input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

#steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/train_all_16K_13dim
#steps/compute_cmvn_stats.sh data/train_all_16K_13dim
#utils/fix_data_dir.sh data/train_all_16K_13dim

#steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" --stage 3 data/train_all_16K_13dim data/lang_tri5b exp/tri5b exp/tri5b_ali || exit 1;

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage -1 \
    --cmd="$run_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=false \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang_tri5b \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
