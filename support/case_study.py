import os
import json
import numpy as np


def main():
    log_dir = './log/adv_dc_visual2'
    for filename in os.listdir(log_dir):
        cont = False
        for cand_imgname in [202175131, 539676201, 4373983146]:
            if str(cand_imgname) in filename:
                cont = True
        if not cont:
            continue
        log = json.load(open(os.path.join(log_dir, filename), 'r'))
        print('='*100)
        print(log)
        # print('='*100)
        # print(log['idx'], log['imgname'])
        # print("| gt_text: ", log['gt_text'])
        gt_text = log['gt_text']
        syn_texts = log['syn_texts']
        disc_ps = log['disc_p']
        correction_texts = log['correction_texts']
        syn_scores = log['syn_scores']

        i = 0
        for syn_score, syn_text, disc_p, correction_text in zip(syn_scores, syn_texts, disc_ps, correction_texts):
            syn_idt = (np.array(syn_text) == np.array(gt_text)).astype(np.int64)
            new_correction_text = []
            for idt, gt_token, correction_token in zip(syn_idt, gt_text, correction_text):
                if idt == 1:
                    new_correction_text.append(gt_token)
                else:
                    new_correction_text.append(correction_token)
            correction_text = new_correction_text
            # correction_text = (syn_idt * gt_text) + (correction_text * (1-syn_idt))

            correction_idt = (np.array(correction_text) == np.array(gt_text)).astype(np.int64)
            # print(syn_idt, correction_idt)
            if correction_idt.sum() >= syn_idt.sum() + 2: #  and (1 - correction_idt).sum() == 0:
                print(i, syn_score, log['imgname'])
                print(list(zip(gt_text, syn_text, correction_text, disc_p)))

            i += 1


if __name__ == '__main__':
    main()
