import sys
import codecs
import numpy as np
import os

def EditDistance(ref, dec):
    """
    """
    # Calculate the sizes of the strings or arrays
    ref_len, dec_len = len(ref), len(dec)
    # distance
    d = np.zeros((ref_len + 1, dec_len + 1), dtype=int)
    # Initialize
    for i in range(dec_len + 1):
        d[0][i] = i
    for j in range(ref_len + 1):
        d[j][0] = j
    # error, 2: ch & en, 3: ins del sub
    e = np.zeros((ref_len + 1, dec_len + 1, 2, 3), dtype=int)
    # Initialize
    for i in range(1, dec_len + 1):
        # ins
        if 'a' <= dec[i-1][0] <= 'z' or 'A' <= dec[i-1][0] <= 'Z':
            e[0][i][1][0] = i
        else:
            e[0][i][0][0] = i
    for j in range(1, ref_len + 1):
        # del
        if 'a' <= ref[j-1][0] <= 'z' or 'A' <= ref[j-1][0] <= 'Z':
            e[j][0][1][1] = j
        else:
            e[j][0][0][1] = j

    # Get error_ch and error_en: ins del sub
    for i in range(1, ref_len + 1):
        for j in range(1, dec_len + 1):
            add_cost = (dec[j-1] != ref[i-1] and 1 or 0)
            val = 0
            if d[i][j-1] + 1 <= min(d[i-1][j-1] + add_cost, d[i-1][j] + 1):
                val = d[i][j-1] + 1
                # ins
                e[i][j] = e[i][j-1]
                if 'a' <= dec[j-1][0] <= 'z' or 'A' <= dec[j-1][0] <= 'Z':
                    e[i][j][1][0] = e[i][j-1][1][0] + 1
                else:
                    e[i][j][0][0] = e[i][j-1][0][0] + 1
            elif d[i-1][j] + 1 <= min(d[i-1][j-1] + add_cost, d[i][j-1] + 1):
                val = d[i-1][j] + 1
                # del
                e[i][j] = e[i-1][j]
                if 'a' <= ref[i-1][0] <= 'z' or 'A' <= ref[i-1][0] <= 'Z':
                    e[i][j][1][1] = e[i-1][j][1][1] + 1
                else:
                    e[i][j][0][1] = e[i-1][j][0][1] + 1
            else:
                val = d[i-1][j-1] + add_cost
                # sub
                e[i][j] = e[i-1][j-1]
                if 'a' <= ref[i-1][0] <= 'z' or 'A' <= ref[i-1][0] <= 'Z':
                    e[i][j][1][2] = e[i-1][j-1][1][2] + add_cost
                else:
                    e[i][j][0][2] = e[i-1][j-1][0][2] + add_cost
            d[i][j] = val

    # Get word number for ch and en
    total_sen = np.array([0, 0])
    for r in ref:
        if 'a' <= r[0] <= 'z' or 'A' <= r[0] <= 'Z':
            total_sen[1] += 1
        else:
            total_sen[0] += 1

    return total_sen, e[ref_len][dec_len][0], e[ref_len][dec_len][1]


def get_ref_hyp(base_path):
    #
    print("ok")
    best_wer_path = os.path.join(base_path, "best_wer")
    content = ''
    with open(best_wer_path, 'r', encoding='utf-8') as fp1:
        content = fp1.read()
    #print(content.rstrip().split('_')[-2:])
    decode_num = content.rstrip().split('_')[-2]
    penalty_num = content.rstrip().split('_')[-1]
    
    best_hyp_path = os.path.join(base_path, "penalty_" + penalty_num + '/' + decode_num + '.txt')

    print("bast hyp txt:", best_hyp_path)
    return best_hyp_path


def main():
    # reference and decode file
    decode_path = sys.argv[1]
    #dec = sys.argv[2]
    ref = os.path.join(decode_path, 'test_filt.txt')
    dec = get_ref_hyp(decode_path)

    a_f = os.system('sort '+ref+' -o '+ref)
    d_f = os.system('sort '+dec+' -o '+dec)
    print(a_f, d_f)

    print(ref)
    print(dec)
    
    #quit()
    # total num for ch and en
    total = np.array([0, 0])
    # ins, del, sub
    error_ch = np.array([0, 0, 0])
    error_en = np.array([0, 0, 0])
    with codecs.open(ref, 'r', 'utf-8') as fref:
        with codecs.open(dec, 'r', 'utf-8') as fdec:
            for ref_line, dec_line in zip(fref, fdec):
                ref_words = ref_line.strip().split()
                dec_words = dec_line.strip().split()
                assert ref_words[0] == dec_words[0]
                total_sen, error_ch_sen, error_en_sen = EditDistance(ref_words[1:], dec_words[1:])
                total += total_sen
                error_ch += error_ch_sen
                error_en += error_en_sen
#                print(ref_line.encode('utf-8'))
#                print(dec_line.encode('utf-8'))
    # CH results
    wer_ch = 100.0 * sum(error_ch) / max(total[0], 1)
    print(' CH: %%CER %.2f [ %d / %d, %d ins, %d del, %d sub ]'
          % (wer_ch, sum(error_ch), total[0], error_ch[0], error_ch[1], error_ch[2]))
    # EN results
    wer_en = 100.0 * sum(error_en) / max(total[1], 1)
    print(' EN: %%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]'
          % (wer_en, sum(error_en), total[1], error_en[0], error_en[1], error_en[2]))
    # All results
    wer_all = 100.0 * (sum(error_ch) + sum(error_en)) / max(sum(total), 1)
    print('MIX: %%MER %.2f [ %d / %d, %d ins, %d del, %d sub ]'
          % (wer_all, sum(error_en) + sum(error_ch), sum(total),
             error_en[0] + error_ch[0], error_en[1] + error_ch[1], error_en[2] + error_ch[2]))
    print('Result:%.2f%%,%.2f%%,%.2f%%' % (wer_ch, wer_en, wer_all))

    result_report = os.path.join(decode_path, 'best_mer')

    result_content =  'CH: %%CER %.2f [ %d / %d, %d ins, %d del, %d sub ]'% (wer_ch, sum(error_ch), total[0], error_ch[0], error_ch[1], error_ch[2]) + '\n' + ' EN: %%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]'% (wer_en, sum(error_en), total[1], error_en[0], error_en[1], error_en[2]) + '\n' + 'MIX: %%MER %.2f [ %d / %d, %d ins, %d del, %d sub ]'%(wer_all, sum(error_en) + sum(error_ch), sum(total),error_en[0] + error_ch[0], error_en[1] + error_ch[1], error_en[2] + error_ch[2]) + '\n' + 'Result:%.2f%%,%.2f%%,%.2f%%' % (wer_ch, wer_en, wer_all)

    with open(result_report, 'a', encoding="utf-8") as fp3:
        fp3.write(result_content)



if __name__ == '__main__':
     main()
