from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import sacrebleu
from time import time

import torch
import pickle

import pandas as pd

it_en = 'Helsinki-NLP/opus-mt-it-en'
en_it = 'Helsinki-NLP/opus-mt-en-it'

data_path = 'Europarl_10000_42.pkl'

# instatiate dataloader class

class MTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, n_samples):
        self.encodings = encodings
        self.n_samples = n_samples

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.n_samples

#define bleu function
def get_cumulative_bleu(reference_sents, candidate_sents):
    reference_tokens = [word_tokenize(sent) for sent in reference_sents]
    candidate_tokens = [word_tokenize(sent) for sent in candidate_sents]

    weights = ((1,0,0,0), (.5,.5,0,0), (.33,.33,.33,0), (.25,.25,.25,.25))

    scores = []
    for weight in weights:
        for ref_tokens, cand_tokens in zip(reference_tokens, candidate_tokens):
            try:
                bleu_temp = sentence_bleu(ref_tokens, cand_tokens, weights=weight)
                scores.append(bleu_temp)
            except KeyError:
                scores.append(0)

    scores = pd.Series(scores)
    scores = scores.values.reshape(len(weights), len(reference_tokens)).T

    return pd.DataFrame(scores)

def get_bleu(reference_sents, candidate_sents):

    scores = []
    for cand, ref in zip(candidate_sents, reference_sents):
        scores.append(sacrebleu.sentence_bleu(cand, ref).score)

    return pd.Series(scores)

def inference_run(tokenizer, dataloader, model, *args, **kwargs):

    count = 1
    start_time = time()

    model.to('cuda')

    translated_sents = []
    with torch.no_grad():
        for batch in dataloader:
            iter_start = time()

            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')

            translated = model.generate(input_ids, attention_mask=attention_mask, *args, **kwargs)

            # extract translated sentences
            translated_decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

            # add decoded sentences to list
            translated_sents.extend(translated_decoded)

            print("\nCompleted iteration {} of {}. ".format(count, len(it_en_model_dataloader)))
            print("Iteration time: {}.".format(round(time() - iter_start, 2)))
            print("Total inference time: {}.".format(round((time() - start_time) / 60, 2)))

            count += 1

    torch.cuda.empty_cache()
    return translated_sents


#load pickled data
f = open(data_path, 'rb')
sentences = pickle.load(f)

# prepare italian -> english data
it_en_tokenizer = MarianTokenizer.from_pretrained(it_en)
encoded_sentences_italian = it_en_tokenizer(sentences['italian'], return_tensors='pt', padding=True)

#instantiate dataset and dataloader for italian -> english models
it_en_model_dataset = MTDataset(encoded_sentences_italian, len(sentences['sent_id']))
it_en_model_dataloader = DataLoader(it_en_model_dataset, batch_size=25)

# instantiate italian -> english models
it_en_greedy_model = MarianMTModel.from_pretrained(it_en, output_scores=True, num_beams=1)
it_en_beam_model = MarianMTModel.from_pretrained(it_en, output_scores=True, num_beams=30)

# run inference on italian -> english models
translated_sents_greedy_en = inference_run(it_en_tokenizer, it_en_model_dataloader, it_en_greedy_model)
translated_sents_beam_en = inference_run(it_en_tokenizer, it_en_model_dataloader, it_en_beam_model)


###################### change translation direction
# prepare english -> italian data
en_it_tokenizer = MarianTokenizer.from_pretrained(en_it)
encoded_sentences_english = en_it_tokenizer(sentences['english'], return_tensors='pt', padding=True)

#instantiate dataset and dataloader for english -> italian models
en_it_model_dataset = MTDataset(encoded_sentences_english, len(sentences['sent_id']))
en_it_model_dataloader = DataLoader(en_it_model_dataset, batch_size=25)

# instantiate english -> italian models
en_it_greedy_model = MarianMTModel.from_pretrained(en_it, output_scores=True, num_beams=1)
en_it_beam_model = MarianMTModel.from_pretrained(en_it, output_scores=True, num_beam=30)

translated_sents_greedy_it = inference_run(en_it_tokenizer, en_it_model_dataloader, en_it_greedy_model)
translated_sents_beam_it = inference_run(en_it_tokenizer, en_it_model_dataloader, en_it_beam_model)


sentences_df = pd.DataFrame(sentences)

output_df = pd.DataFrame({
    'sent_id': sentences['sent_id'],
    'it_en_greedy': translated_sents_greedy_en,
    'it_en_beam': translated_sents_beam_en,
    'en_it_greedy': translated_sents_greedy_it,
    'en_it_beam': translated_sents_beam_it
})

results_df = sentences_df.merge(output_df, on='sent_id')
results_df['it_en_match'] = results_df['it_en_greedy'] == results_df['it_en_beam']
results_df['en_it_match'] = results_df['en_it_greedy'] == results_df['en_it_beam']


f = open('MT_beam_df_'+results_df, "wb")
pickle.dump(results_df, f)
f.close()

it_en_greedy_bleu = get_cumulative_bleu(results_df['english'], results_df['it_en_greedy'])
it_en_greedy_bleu.columns = ['it_en_greedy_bleu1', 'it_en_greedy_bleu2', 'it_en_greedy_bleu3', 'it_en_greedy_bleu4']

it_en_beam_bleu =  get_cumulative_bleu(results_df['english'], results_df['it_en_beam'])
it_en_beam_bleu.columns = ['it_en_beam_bleu1', 'it_en_beam_bleu2', 'it_en_beam_bleu3', 'it_en_beam_bleu4']

en_it_greedy_bleu = get_cumulative_bleu(results_df['italian'], results_df['en_it_greedy'])
en_it_greedy_bleu.columns = ['en_it_greedy_bleu1', 'en_it_greedy_bleu2', 'en_it_greedy_bleu3', 'en_it_greedy_bleu4']

en_it_beam_bleu = get_cumulative_bleu(results_df['italian'], results_df['en_it_beam'])
en_it_beam_bleu.columns = ['en_it_beam_bleu1', 'en_it_beam_bleu2', 'en_it_beam_bleu3', 'en_it_beam_bleu4']

bleu_df = pd.concat((it_en_greedy_bleu, it_en_beam_bleu, en_it_greedy_bleu, en_it_beam_bleu), axis=1)
#bleu_df = bleu_df.astype(float)

it_en_greedy_sacrebleu = get_bleu(results_df['it_en_greedy'], results_df['english'])
it_en_beam_sacrebleu = get_bleu(results_df['it_en_beam'], results_df['english'])
en_it_greedy_sacrebleu = get_bleu(results_df['en_it_greedy'], results_df['italian'])
en_it_beam_sacrebleu = get_bleu(results_df['en_it_beam'], results_df['italian'])

sacrebleu_df = pd.DataFrame({
    'it_en_greedy_sacrebleu ': it_en_greedy_sacrebleu,
    'it_en_beam_sacrebleu': it_en_beam_sacrebleu,
    'en_it_greedy_sacrebleu': en_it_greedy_sacrebleu,
    'en_it_beam_sacrebleu': en_it_beam_sacrebleu
})


final_df = pd.concat([results_df, bleu_df], axis=1)


f = open('MT_final_nlt_bleu_df.pkl', "wb")
pickle.dump(final_df, f)
f.close()

