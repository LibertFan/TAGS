import os
import json

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# from nlp import load_dataset
from tqdm import tqdm
import numpy as np
import string


def load_data():
    puncs = list(set(list(string.punctuation)) - set([',', '.', '-', '?', '!', '\'']))
    puncs = [punc + ' ' for punc in puncs]

    def bpe2token(bpes):
        text = ' '.join(bpes).replace(' ##', '').replace(" ,", ',').replace(" .", ".").replace(' - ', '-').\
            replace(' ?', '?').replace(' !', '!').replace(' \' ', '\'')
        for punc in puncs:
            text = text.replace(punc, ' ')
        text = text.strip()
        return text

    log_dir = './log/adv_dc_visual2'
    dataset_gt_texts, dataset_syn_texts, dataset_correct_texts = [], [], []

    max_num = 1
    for filename in tqdm(os.listdir(log_dir)):
        log = json.load(open(os.path.join(log_dir, filename), 'r'))
        gt_text = log['gt_text'][1:-1]
        syn_texts = [text[1:-1] for text in log['syn_texts'][:max_num]]
        correct_texts = [text[1:-1] for text in log['correction_texts'][:max_num]]
        # print("| gt_text: ", gt_text)
        # print("| syn_text: ", syn_texts[0])
        # print("| correct_text: ", correct_texts[0])
        dataset_gt_texts.append(bpe2token(gt_text))
        dataset_syn_texts.extend([bpe2token(syn_text) for syn_text in syn_texts])
        dataset_correct_texts.extend([bpe2token(correct_text) for correct_text in correct_texts])
    return dataset_gt_texts, dataset_syn_texts, dataset_correct_texts


def main():
    # device = 'cuda'
    # from transformers import GPT2Tokenizer, GPT2LMHeadModel
    # # Load pre-trained model (weights)
    # with torch.no_grad():
    #     model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    #     model.eval()
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = 'cuda'
    model_id = 'gpt2-large'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # text = "From large scale power generators to the basic cooking in our homes, fuel is essential for all of these to happen and work."
    # text = "A woman plays with finger puppets as a small child in a costume walks by."
    # # "From large scale power generators to the basic cooking in our homes, fuel is essential for all of these to happen and work."
    # input_ids = torch.tensor(tokenizer([text])["input_ids"]).to(device)
    # target_ids = input_ids.clone()
    # with torch.no_grad():
    #     log_likelihood = model(input_ids, labels=target_ids)[0]
    #     ppl = torch.exp(log_likelihood)
    #     print(np.exp(log_likelihood.data.cpu().numpy()))
    #     print('1: ', log_likelihood, ppl)
    #
    # raise Exception

    # test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

    # max_length = model.config.n_positions
    # stride = 512

    dataset_gt_texts, dataset_syn_texts, dataset_correct_texts = load_data()
    for i, dataset_texts in enumerate([dataset_gt_texts, dataset_syn_texts, dataset_correct_texts]):
        lls = []
        end_loc = 0
        lls2 = []
        end_loc2 = 0
        lls_save = []
        ppls = []
        for text in tqdm(dataset_texts):
            # print(" text: ", text, type(text))
            # print('| text: ', text[0], text[1], text[-1])
            input_ids = torch.tensor(tokenizer([text])["input_ids"]).to(device)
            # print('| ', text, input_ids)
            trg_len = input_ids.size(-1)
            end_loc += 1
            end_loc2 += trg_len
            # print(type(input_ids))
            # print(input_ids)
            # print('| input_ids: ', input_ids.size(), input_ids)
            input_ids = input_ids
            target_ids = input_ids.clone()
            # print('| input_ids', input_ids.size(), target_ids.size())
            # print(input_ids.size(), target_ids.size())
            # target_ids[-1] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                ll = outputs[0].item()
                log_likelihood = outputs[0]
                ppl = torch.exp(log_likelihood)
                if ppl > 200.0:
                    print('| large ppl: ', text)
                # print(ll, ppl)

                ppls.append(ppl)
                # print('| log_likelihood: ', log_likelihood, ppl)
                lls.append(log_likelihood)
                lls2.append(log_likelihood * trg_len)
                lls_save.append(ppl.item())

        ppl_mean = torch.stack(ppls).mean()
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        ppl2 = torch.exp(torch.stack(lls2).sum() / end_loc2)
        print('| ppl: ', ppl_mean.item(), ppl.item(), ppl2.item())

        save_path = os.path.join('log', 'dataset_{}.json'.format(i))
        with open(save_path, 'w') as fw:
            json.dump(lls_save, fw)
        fw.close()
        print('ppl: ', ppl)

    # for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    #     begin_loc = max(i + stride - max_length, 0)
    #     end_loc = min(i + stride, encodings.input_ids.size(1))
    #     trg_len = end_loc - i    # may be different from stride on last loop
    #     input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    #     target_ids = input_ids.clone()
    #     target_ids[:,:-trg_len] = -100
    #
    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=target_ids)
    #         log_likelihood = outputs[0] * trg_len

if __name__ == '__main__':
    main()
