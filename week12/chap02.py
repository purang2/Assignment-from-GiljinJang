# ID: 2021220699
# NAME: Eunchan Lee 
# File name: chap02.py
# Platform: Python 3.9 on Windows 10 Spyder5
# Required Package(s): sys numpy matplotlib sklearn pandas

# -*- coding: utf-8 -*-



%matplotlib inline

import numpy as np 
import sys
import matplotlib.pyplot as plt
from dataset import ptb
sys.path.append('..')
#from common.util import preprocess 

import pandas as pd
import matplotlib as mpl
from sklearn.manifold import TSNE


text = 'You say goodbye and I say hello.'
text = text.lower().replace('.', ' .')
print("'", text, "'")

words = text.split(' ')
print(words)

word_to_id = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id

id_to_word = {id_:  word for word, id_ in word_to_id.items()}

print(id_to_word)
print(word_to_id)


print(id_to_word[1])
print(word_to_id['hello'])

# equivalent to the above, using set
# difference: 위의 것은 나온 순서대로, 아래 것은 순서가 달라짐
wordset = set(words) # remove duplicates
word_to_id_2 = dict(zip(wordset,list(range(len(wordset)))))
id_to_word_2 = {id_: word for word, id_ in word_to_id_2.items()}
print(wordset)
print(id_to_word_2)
print(word_to_id_2)


corpus = np.array([word_to_id[word] for word in words])


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word



text = 'Hi My name is James Louis Eunchan and you say Hi Eunchan'
corpus, word_to_id, id_to_word = preprocess(text)


corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size),dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i # left window_size
            right_idx = idx + i # right window_size

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

print(enumerate(corpus))

for word_id in corpus: print(word_id)
for idx, word_id in enumerate(corpus): print(idx, word_id)



window_size = 1 # 주변 1개
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size)

print(C)

def cos_similarity(x, y, eps=1e-8):
    # epsilon 값을 추가해,
    # 0으로 나누기 오류가 나는 것을 막아줌
    nx = x / np.sqrt(np.sum(x**2) + eps) # x의 정규화
    ny = y / np.sqrt(np.sum(y**2) + eps) # y의 정규화
    
    return np.dot(nx, ny)


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]

print(cos_similarity(c0,c1))


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query}(을)를 찾을 수 없습니다.')
        return 
    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count = 0
    for i in (-1* similarity).argsort():
        if id_to_word[i] == query:
            continue 
        print(f' {id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return 
        
        
text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size= len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

def ppmi(C, verbose=False, eps=1e-8):

    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[i]*S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print(f'{(100*cnt/total):.2f} 완료')
    return M


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
np.set_printoptions(precision=3) # 유효 자릿수를 세 자리로 표시
print('Co-occurrence Matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

print(most_similar('you', word_to_id, id_to_word, W, top=5))



corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
# SVD
U, S, V = np.linalg.svd(W)



print(C[0]) # 동시발생 행렬
print(W[0]) # PPMI 행렬
print(U[0]) # SVD
# 2차원으로 차원 축소하기
print(U[0, :2])



# 플롯
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()

corpus, word_to_id, id_to_word = ptb.load_data('train')

print('말뭉치 크기:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[1]:', id_to_word[1])
print('id_to_word[2]:', id_to_word[2])
print()
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy'])
print("word_to_id['lexus']:", word_to_id['lexus'])


window_size = 2
wordvec_size = 100 

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('Create Co-Matrix ...')
C = create_co_matrix(corpus,vocab_size, window_size)

print('PPMI 계산...')
W = ppmi(C, verbose=True)



try: 
    from sklearn.utils.extmath import randomized_svd 
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,random_state=None)
except:
    U, S, V = np.linalg.svd(W)


word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']

for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(U[1000:1100,:])   

vocab = list(id_to_word.values())


df = pd.DataFrame(X_tsne, index=vocab[1000:1100], columns=['x', 'y'])


print(df.head(10))

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()



