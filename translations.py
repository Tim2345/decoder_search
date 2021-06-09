from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

import torch
import pickle

import pandas as pd

it_en = 'Helsinki-NLP/opus-mt-it-en'
en_it = 'Helsinki-NLP/opus-mt-en-it'

# instatiate dataloader class

class MTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, n_samples):
        self.encodings = encodings
        self.n_samples = n_samples
        #self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.n_samples


#load pickled data
f = open('Europarl_20_42.pkl', 'rb')
sentences = pickle.load(f)

model_name = it_en
it_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
encoded_sentences_italian = it_en_tokenizer(sentences['italian'], return_tensors='pt', padding=True)

#data_loader = DataLoader(encoded_sentences_italian, shuffle=True, batch_size=10)
it_en_model_dataset = MTDataset(encoded_sentences_italian, len(sentences['sent_id']))
it_en_model_dataloader = DataLoader(it_en_model_dataset, batch_size=10)

#encoded_sentences_english = en_it_tokenizer(sentences['english'], return_tensors='pt', padding=True)

it_en_model = MarianMTModel.from_pretrained(it_en, output_scores=True)

it_en_model.to('cuda')
translated_sents_greedy_it = []
translated_sents_beam_it = []

for batch in tqdm(it_en_model_dataloader):

    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')

    translated_greedy = it_en_model.generate(input_ids, attention_mask=attention_mask, num_beams=1)
    translated_beam = it_en_model.generate(input_ids, attention_mask=attention_mask, num_beams=30)

    # extract translated sentences
    translated_sents_greedy_it.extend([it_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated_greedy])
    translated_sents_beam_it.extend([it_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated_beam])


torch.cuda.empty_cache()



model_name = en_it
en_it_tokenizer = MarianTokenizer.from_pretrained(model_name)
encoded_sentences_english = en_it_tokenizer(sentences['english'], return_tensors='pt', padding=True)

#data_loader = DataLoader(encoded_sentences_italian, shuffle=True, batch_size=10)
en_it_model_dataset = MTDataset(encoded_sentences_english, len(sentences['sent_id']))
en_it_model_dataloader = DataLoader(en_it_model_dataset, batch_size=10)

#encoded_sentences_english = en_it_tokenizer(sentences['english'], return_tensors='pt', padding=True)

en_it_model = MarianMTModel.from_pretrained(en_it, output_scores=True)

en_it_model.to('cuda')
translated_sents_greedy_en = []
translated_sents_beam_en = []

for batch in tqdm(en_it_model_dataloader):

    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')

    translated_greedy = en_it_model.generate(input_ids, attention_mask=attention_mask, num_beams=1)
    translated_beam = en_it_model.generate(input_ids, attention_mask=attention_mask, num_beams=30)

    # extract translated sentences
    translated_sents_greedy_en.extend([en_it_tokenizer.decode(t, skip_special_tokens=True) for t in translated_greedy])
    translated_sents_beam_en.extend([en_it_tokenizer.decode(t, skip_special_tokens=True) for t in translated_beam])


sentences_df = pd.DataFrame(sentences)

output_df = pd.DataFrame({
    'sent_id': sentences['sent_id'],
    'it_en_greedy': translated_sents_greedy_it,
    'it_en_beam': translated_sents_beam_it,
    'en_it_greedy': translated_sents_greedy_en,
    'en_it_beam': translated_sents_beam_en
})

results_df = sentences_df.merge(output_df, on='sent_id')
results_df['it_en_match'] = results_df['it_en_greedy'] == results_df['it_en_beam']
results_df['en_it_match'] = results_df['en_it_greedy'] == results_df['en_it_beam']


def get_cumulative_bleu(reference_sents, candidate_sents):
    reference_tokens = [word_tokenize(sent) for sent in reference_sents]
    candidate_tokens =

sentence_bleu(word_tokenize(results_df['english'][1]), word_tokenize(results_df['it_en_greedy'][1]), weights=(0,1,0,0))

def get_bleu_scores(target_sents)
results_df['it_en_greedy_bleu']