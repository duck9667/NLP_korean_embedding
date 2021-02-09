
# 3장 한국어 전처리
## 3.1. 데이터 확보
### 3.1.1. 한국어 위키백과


```python
# from preprocess import dump.tokenize
# !python dump.py
from gensim.corpora import WikiCorpus, Dictionary
from gensim.utils import to_unicode

in_f = "/notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2" 
out_f = "/notebooks/embedding/data/processed/processed_wiki_ko.txt"
output = open(out_f, 'w')
wiki = WikiCorpus(in_f, tokenizer_func=tokenize, dictionary=Dictionary())
i = 0
for text in wiki.get_texts():
    output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
    i = i + 1
    if (i % 10000 == 0):
        print('Processed ' + str(i) + ' articles')
output.close()
print('Processing complete!')
```

    Processed 10000 articles
    Processed 20000 articles
    Processed 30000 articles
    Processed 40000 articles
    Processed 50000 articles
    Processed 60000 articles
    Processed 70000 articles
    Processed 80000 articles
    Processed 90000 articles
    Processed 100000 articles
    Processed 110000 articles
    Processed 120000 articles
    Processed 130000 articles
    Processed 140000 articles
    Processed 150000 articles
    Processed 160000 articles
    Processed 170000 articles
    Processed 180000 articles
    Processed 190000 articles
    Processed 200000 articles
    Processed 210000 articles
    Processed 220000 articles
    Processed 230000 articles
    Processed 240000 articles
    Processed 250000 articles
    Processed 260000 articles
    Processed 270000 articles
    Processed 280000 articles
    Processed 290000 articles
    Processed 300000 articles
    Processed 310000 articles
    Processed 320000 articles
    Processed 330000 articles
    Processed 340000 articles
    Processed 350000 articles
    Processed 360000 articles
    Processing complete!


### 3.1.2. KorQuAD
LG CNS가 공개한 데이터셋으로 문서 중 일부 문단으로부터 파생가능한 질문과 답변의 쌍으로 구성되어있다.


```python
import json

corpus_fname = "data/raw/KorQuAD_v1.0_train.json"
output_fname = "data/processed_korquad_train.txt"

with open(corpus_fname) as f1, open(output_fname, 'w', encoding = 'utf-8') as f2 :
    dataset_json = json.load(f1)
    dataset = dataset_json['data']
    for article in dataset :
        w_lines = []
        for paragraph in article['paragraphs'] :
            w_lines.append(paragraph['context'])
            for qa in paragraph['qas'] :
                q_text = qa['question']
                for a in qa['answers'] :
                    a_text = a['text']
                    w_lines.append(q_text + ' ' + a_text)

                for line in w_lines :
                    f2.writelines(line + '\n')
```

### 3.1.3. 네이버 영화 리뷰 말뭉치
네이버 영화 리뷰를 수집한 말뭉치로 긍정/부정으로 라벨링되어 있다.


```python
corpus_path = "data/raw/ratings.txt"
output_fname = "data/processed/processed_ratings.txt"
with_label = False

with open(corpus_path, 'r', encoding = 'utf-8') as f1, open(output_fname, 'w', encoding = 'utf-8') as f2 :
    next(f1)
    for line in f1 :
        _, sentence, label = line.strip().split('\t')
        if not sentence : continue
        if with_label :
            f2.writelines(sentence + '\u241E' + label + '\n')
        else :
            f2.writelines(sentence + '\n')
```

## 3.2. 지도 학습 기반 형태소 분석
한국어는 조사와 어미갈 발달한 교착어로 섬세한 전처리가 필요하다.

### 3.2.1. KoNLPy 사용법 


```python
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

def get_tokenizer(tokenizer_name) :
    if tokenizer_name == "komoran" :
        tokenizer = Komoran()
    elif tokenizer_name == "okt" :
        tokenizer = Okt()
    elif tokenizer_name == "mecab" :
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum" :
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma" :
        tokenizer = Kkma()
    else :
        tokenizer = Mecab()
    return tokenizer
```


```python
tokenizer = get_tokenizer("komaran")
tokenizer.morphs("아버지가방에들어가신다")
tokenizer.pos("아버지가방에들어가신다")
```




    [('아버지', 'NNG'),
     ('가', 'JKS'),
     ('방', 'NNG'),
     ('에', 'JKB'),
     ('들어가', 'VV'),
     ('신다', 'EP+EC')]



### 3.2.3. Khaiii 사용법


```python
from khaiii import KhaiiiApi
tokenizer = KhaiiiApi()
```


```python
# 형태소 분석
data = tokenizer.analyze('아버지가방에들어가신다')
tokens = []
for word in data :
    tokens.extend([str(m).split("/")[0] for m in word.morphs])
tokens
```




    ['아버지', '가', '방', '에', '들어가', '시', 'ㄴ다']




```python
# 품사 정보 확인
data = tokenizer.analyze('아버지가방에들어가신다')
tokens = []
for word in data :
    tokens.extend([str(m) for m in word.morphs])
data = tokenizer.analyze('아버지가방에들어가신다')
```




    ['아버지/NNG', '가/JKS', '방/NNG', '에/JKB', '들어가/VV', '시/EP', 'ㄴ다/EC']



### 3.2.4. 은전한닢에 사용자 사전 추가하기
오리전자 같은 단어는 사용자 사전에 없는 단어로 이 경우, 하나의 단어로 토큰화될 수 있도록 강제해야한다.


```python
from konlpy.tag import Mecab
tokenizer = Mecab()
tokenizer.morphs('오리전자 텔레비전 정말 좋네요')
```




    ['오리', '전자', '텔레비전', '정말', '좋', '네요']




```python
from konlpy.tag import Mecab
tokenizer = Mecab()
tokenizer.morphs('오리전자 텔레비전 정말 좋네요')
```




    ['오리전자', '텔레비전', '정말', '좋', '네요']



## 3.3. 비지도 학습 기반 형태소 분석
앞서 살펴본 기법은 전문가가 직접 태깅한 데이터를 기반으로 학습한 것, 이와 달리 비지도 학습 기법은 모델 스스로 데이터의 패턴을 학습하게 함으로써 형태소를 분석한다.

### 3.3.1. soynlp 형태소 분석기
패턴을 스스로 학습하는 비지도 학습을 지향하므로 동질적인 문서 집합에서 잘 작동한다. 해당 분석기는 데이터의 통계량을 확인해 만든 단어 점수 표로 작동한다. 단어 점수는 크게 `응집 확률`과 `브랜칭 엔트로피` 활용한다.
- 응집확률 : 단어 노출 빈도가 높을 수록 높음
- 브랜칭 엔트로피 : 해당 문자열 앞뒤로 조사 또는 어미 혹은 다른 단어가 등장하는 경우


```python
# 모델 학습 
from soynlp.word import WordExtractor

corpus_fname = '/notebooks/embedding/data/processed/processed_ratings.txt'
model_fname = '/notebooks/embedding/data/processed/soyword.model'

sentences = [sent.strip() for sent in open(corpus_fname, 'r').readlines()]
word_extractor = WordExtractor(min_frequency = 100,
                              min_cohesion_forward = 0.05,
                              min_right_branching_entropy = 0.0)
word_extractor.train(sentences)
word_extractor.save(model_fname)
```


```python
# 형태소 부석
import math
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

model_fname = '/notebooks/embedding/data/processed/soyword.model'

word_extractor = WordExtractor(min_frequency = 100,
                              min_cohesion_forward = 0.05,
                              min_right_branching_entropy = 0.0)

word_extractor.load(model_fname)
scores = word_extractor.word_scores()
scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
tokenizer = LTokenizer(scores=scores)
tokens = tokenizer.tokenize("애비는 종이었다")
```

    all cohesion probabilities was computed. # words = 6130
    all branching entropies was computed # words = 123575
    all accessor variety was computed # words = 123575


### 3.3.2. 구글 센텐스피스 

구글에서 공개한 비지도 학습 기반 형태소 분석 패키지다. `BEP`의 기본 원리는 말뭉치에서 가장 많이 등장한 문자열을 병합해 문자열을 압축하는 것이다. `BERT` 모델은 BPE로 학습한 어휘 집합을 쓴다. 하지만 사용 전 일부 후처리가 필요하다. 언더바 `_` 문자를 `##`로 바꾸고 스페셜 토큰을 추가한다.


```python
import sentencepiece as spm
train = """--input=/notebooks/embedding/data/processed/processed_wiki_ko.txt \
           --model_prefix=sentpiece \
           --vocab_size=32000 \
           --model_type=bpe --character_coverage=0.9995"""
```


```python
from models.bert.tokenization import FullTokenizer

vocab_fname = "/notebooks/embedding/data/processed/bert.vocab"
tokenizer = FullTokenizer(vocab_file = vocab_fname, do_lower_case = False)

tokenizer.tokenize('집에좀 가자')
```




    ['집에', '##좀', '가자']



### 3.3.3. 띄어쓰기 교정
soynlp에서 띄어쓰기 교정 모듈을 제공한다. 자연어 처리시 띄어쓰기 교정을 하면 분석의 품질이 개선된다.  
전처리시 `띄어쓰기 교정`, `형태소 분석`, `불용어 제거` 등을 꼭 염두해야 한다.


```python
from soyspacing.countbase import CountSpace

corpus_fname = "/notebooks/embedding/data/processed/processed_ratings.txt"
model_fname = "/notebooks/embedding/data/processed/space-correct.model"

model = CountSpace()
model.train(corpus_fname)
model.save_model(model_fname, json_format = False)
```

    all tags length = 1166880 --> 165338, (num_doc = 199991)


```python
model.correct("어릴때보고 지금다시봐도 재밌어요")
```




    ('어릴때 보고 지금 다시봐도 재밌어요', [0, 0, 1, 0, 1, 0, 1, 0, None, 0, 1, 0, 0, 0, 1])


