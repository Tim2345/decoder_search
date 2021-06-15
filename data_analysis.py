import pandas as pd
import sacrebleu
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

data_name = 'MT_final_bleu_sacrebleu_df.pkl'

data = pd.read_pickle(data_name)

def get_word_counts(df, col_names):
    return {col: [len(word_tokenize(sent)) for sent in df[col]] for col in col_names}

word_len_dict = get_word_counts(data, ['italian', 'english', 'it_en_greedy',
                                       'it_en_beam', 'en_it_greedy', 'en_it_beam'])

word_len_df = pd.DataFrame(word_len_dict)

it_en_word_len = word_len_df[~data['it_en_match']]
en_it_word_len = word_len_df[~data['en_it_match']]

it_en_word_len.mean()
en_it_word_len.mean()

sacrebleu.corpus_bleu(list(data['english']), [list(data['it_en_beam'])])
sacrebleu.corpus_bleu(list(data['english']), [list(data['it_en_greedy'])])


sacrebleu.corpus_bleu(list(data['italian']), [list(data['en_it_greedy'])])
sacrebleu.corpus_bleu(list(data['italian']), [list(data['en_it_beam'])])

# get proportion of translations that do not match betweeen greedy and beam search
1-data['it_en_match'].sum()/data.shape[0]
1-data['en_it_match'].sum()/data.shape[0]

# get proportion of beam search sentences longer than greedy ones when there are differences in translated sentences
pd.Series(it_en_word_len['it_en_beam']>it_en_word_len['it_en_greedy']).sum()/it_en_word_len.shape[0]
pd.Series(it_en_word_len['en_it_beam']>it_en_word_len['en_it_greedy']).sum()/it_en_word_len.shape[0]
# get proportion of beam search sentences longer than greedy ones when there are differences in translated sentences
pd.Series(it_en_word_len['it_en_beam']==it_en_word_len['it_en_greedy']).sum()/it_en_word_len.shape[0]
pd.Series(it_en_word_len['en_it_beam']==it_en_word_len['en_it_greedy']).sum()/it_en_word_len.shape[0]
#get proportion of sentences
pd.Series(it_en_word_len['it_en_beam']<it_en_word_len['it_en_greedy']).sum()/it_en_word_len.shape[0]
pd.Series(it_en_word_len['en_it_beam']<it_en_word_len['en_it_greedy']).sum()/it_en_word_len.shape[0]
