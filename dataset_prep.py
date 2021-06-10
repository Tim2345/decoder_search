import random
import pickle

from operator import itemgetter

#define number of sentences in final dataset
n_sents = 10000
# define max length of sentences in final dataset
max_len = 100

# read in italian data
f = open('europarl-v7.it-en.it', 'r', encoding='utf-8')
it_text = f.read()
f.close()

# read in english data
f = open('europarl-v7.it-en.en', 'r', encoding='utf-8')
en_text = f.read()
f.close()

it_new = it_text.split('\n')
en_new = en_text.split('\n')

# eyeball check that the sentences are aligned
n = 100
for it, en  in zip(it_new[:n], en_new[:n]):
    print('\nItalian:')
    print(it)
    print('English:')
    print(en)


# randomly sample sentences that are no longer than 100 tokens
count = 0
valid = []
for it, en in zip(it_new, en_new):
    if len(it.split())<max_len and len(it.split())<max_len:
        valid.append(count)
    count += 1

seed = 42
random.seed(seed)

indices = sorted(random.sample(valid, n_sents))

# finalize texts
final_italian = itemgetter(*indices)(it_new)
final_english = itemgetter(*indices)(en_new)

# eyeball check that the sentences are aligned
n = 100
for it, en  in zip(final_italian[:n], final_english[:n]):
    print('\nItalian:')
    print(it)
    print('English:')
    print(en)

# put into dictionary
sentence_dict = {'sent_id': list(range(n_sents)),
                 'italian': final_italian,
                 'english': final_english}

#pickle the dictionary

f = open("Europarl_{}_{}.pkl".format(n_sents, seed), "wb")
pickle.dump(sentence_dict, f)
f.close()



