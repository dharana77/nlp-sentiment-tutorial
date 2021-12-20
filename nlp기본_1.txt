Nlp 스터디 1.

part1 에서는 입문자를 위한 
백오브 워드를 소개하는 튜토리얼

코드
https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb#scrollTo=kSCIe6xShOM7

KOBERT
https://www.dinolabs.ai/m/271?category=1203530


Naver Sentiment Movie Corpus

네이버 
이중분류 예시 코드(Naver Sentiment Analysis Fine-Tuning with pytorch)

채널 키워드?
혹은 채널 최적화하여 데이터를 뽑아내는게


2에서는 워드백터
3에서는 워드벡터와 함께 k-means로 단어를 군집화
4에서는 1-3까지 진행이 되었던 딥러닝과 딥러닝이 아닌 메소드를 비교해보는 튜토리얼

Pandas, numpy
scikit-learn
scipy
Beautiful soups… 등 설치되어 있어야

딥러닝이 무엇인지도 자세히 소개가 되어있습니다.

4개의 데이터셋을 제공하고 있는 sampleSubmissionData와

unlabeledTrainData ..

labeledTrainData.tsv ..
unlabeldTrainData.tsv

  nlp와 bag of word of meets bags of popcorn이라는 게 무엇인지

형태소 분석이라든지,
품사부착,
구절 단위 분석,
구문 분석 등이 있는데요,

python natk? 사용해서 해결할 예정인데요,

sentiment analysis of movie reviews 라는 경진대회가 있는데요,
로튼 토마토 데이터셋을 이용해서 사용합니다.

최근에 진행되고 있는 경진대회중에
악플을 선별하는 경진대회가 있구요,
 Toxic Comment Classification Challenge ( Identify and classify toxic online comments)

Spooky author identification이라는 경진대회는 작성자 분류 경진대회?
시즌별

바인딩 되지 않는 책들을 묶으려고 책들을 정리했는데 그 책을 실수로 엎어버림.
바인딩 되지 않은 문장들이 어떤 작가의 문장들인지 선별하는 것.

튜토리얼의 간략한 개요를 보면
초보자를 대상으르ㅗ 기본 자연어 처리를 배우고.,
파트 2,3 에서는 
워드투벡터를 사용하여 모델을 학습시키는 방법, 감정분석과 단어벡터를 사용하는 방법을
보고,
3에서는 k-means 알고리즘을 사용하여 군집화를 해볼 예쩡,

그리고 감정분석 데이터 세트를 통해 자연어처리가 어떤 것인지 보는 목표 달성.

ROC 커브라는 평가 메트릭을 사용하는데
ROC커브가 무엇인지 살펴보고 갈게요
TPR은 민감도, True Positive Rate
암환자를 진찰해서 암이라고 진단함.

특이도 FPR은 암환자가 아닌데 암환자라고 진단한 경우.
ROC 커브 및 면적이 1에 가까울수록 (왼쪽 꼭지점에) 가까울수록 성능이 좋은 모델

자연어 텍스트 분석해서 특정 단어를 얼마나 사용했는지, 얼마나 자주 사용했는지, 어떤 종류의 텍스트인지 분류하거나 긍정인지 부정인지에 대한 감정분석, 그리고 어떤 내용인지 요약하는 정보를
얻을 수 있다.

사람뿐만 아니라 컴퓨터에게도 오해의 소지가 있고, 어려운 분야임

Nltk 모듈에 상당수의 NLP 기능이 구현되어 있는데, 코퍼스, 함수와 알고리즘으로 구성되어 있음.
Anaconda 쓸 경우 잘 설치되는데 가상환경 사용시 설치 잘안되는 문제가 있음.

BOW(bag of words)
라 해서 nlp에서 널리 쓰이는 방법이 있는데,
가장 간단하지만 효과적이라 쓰이고 있음.

각 단어가 이 말뭉치에 얼마나 많이 나타나는지 헤아리는 방법.
단어의 출현횟수를 셈.
(텍스트를 담는 가방으로 생각)

그러나 단어의 순서가 완전히 무시된다는 단점이 있음. 예를 들어 의미가 완전히 반대인 
두 문장이 있다고 하면,
it’s bad, not good at all.
it’s good, not bad at all.
백 오브 워드만 사용하는 경우 의미가 반대이지만 동일하게 해석

이를 보완하기 위해 n-gram을 사용하는데 BOW는 하나의 토큰을 사용하지만 n-gram은 n개의 토큰을 사용할 수 있도록 한다.


——————————————————————


https://www.kaggle.com/c/word2vec-nlp-tutorial

bag of words meets bags of popcorn
튜토리얼의
part1

nlp에 대한 간략한 소개와, pandas로 데이터를 불러오고, 데이터를 정제하고,
벡터화해주고, 랜덤포레스트를 통해서 positive인지 negative인지 예측하는게 큰 맥락입니다.

데이터에 있는 데이터들을 받아주어야 하고요,
Train_data
id-sentiment-review

TestData.tsv
id-review

트레인 데이터 바탕으로 테스트 데이터의 sentiment예측하는게 목표

MJ라는 단어? Moonwalker is part of biography,

혹시 문워커에 대한 영화 리뷰.?

IMDB 데이터를 사용하고 있습니다. 문워커가 어떠한 영화인지 살펴봤는데 마이클잭슨과
관련된 다큐멘터리인것 같습니다.
구글 번역기로 영화 줄거리를 살펴보았더니, 마이클 잭슨이 안좋은 일을 겪었던 것같고,
리뷰가 긍정일까 부정일까 보았는데 센티멘트가 1로 되어있고
positive한 리뷰였습니다.

pandas를 통해서 데이터를 불러오도록 할게요.

header = 0 은 파일의 첫번째 줄에 열 이름이 있음을 나타내며,
delimeter = \t는 필드가 탭으로 구분되는 것을 의미하며,
Quoting = 3 은 쌍따옴표를 무시하도록 한다.

train = pd.read_csv(‘data/labeledTrainData.tsv’, ..)
test = pd.read)_csv(
train.shape
(25000, 3)
트레인데이터의 크기를 알 수 있어요, (25000, 3) 2만 5천개의 리뷰가 있고, 3개의 컬럼이 있습니다.
train.head() 앞에서 5개만의 리뷰를 볼 수 있고,
train.tail() 뒤에서 5개만의 리뷰를 볼 수 있습니다.

test.columns.values

train.info() 를 찍어보면 다 2만5천개 씩 있고,
sentiment데이터는 int로 되어 있어요,

train.describe()를 해보면,
센티멘트의 count는 2만 5천개,
평균값, 중간값 등을 알수 있어요.

train[’sentiment’].value_counts()
1 12500
0 12500 으로 똑같이 나누어져 있어요.

Train[‘review’][0][:700] 
html태그가 섞여있기 때문에 이를 정제해줄 필요가 있음.
———————————————

NLP 데이터 전처리

기계가 텍스트를 이해할 수 있도록 텍스트를 정제해 준다.
신호와 소음을 구분한다. 
아웃라이어데이터로 인한 오버피팅을 제거한다.

1.뷰티풀 숩을 통해 HTML 태그르 제거한다.
정규표현식으로 알파벳 이외의 문자를 공백으로 치환
NLTK데이터를 사용해 불용어(Stopword)를 제거 - 불용어라는 i my me같이 자주 등장하는 단어이지만 의미가 많지는 않은 단어 의미
어간추출(스테밍) 과 음소표기법의 개념을 이해하고 스노우볼 스테머를 통해 어간을 추출

텍스트 데이터 전처리 이해하기
한국어의 경우는 어떻게 하는가?
KoNLpy 를 사용하구요, 트위터 형태소 분석기를 통해 처리.

정규화 normalization(입니닼ㅋㅋ ->입니다 ㅋㅋ, 샤릉해 ->사랑해)
토큰화 (명사 등..)
어근화 (입니다->이다)
어구 추출(한국어, 처리,예시, 처리하는 예시)

import re
#정규표현식을 사용해서 특수문자를 제거
#소문자와 대문자가 아닌 것은 공백으로 대체
Letters_only = re.sub(‘[^a-zA-Z]’,  ‘ ‘, example1.get_text())
letteres_only[:700]
소문자 a-z거나 A-Z가 아니라면 공백으로 대체

Lower_case = letters_only.lower()
#모두 소문자로 변환한다.
#문자를 나눈다. => 토큰화
words = lower_case.split()
print(Len(words))
words[:10]


불용어 제거(Stopward Removal)
일반적으로 코퍼스에서 자주 나타나는 단어는 학습 모델로서 학습이나 예측 프로세스에 실제로 기여하지 않아 다른 텍스트와 구별하지 못한다. 예를들어 조사, 접미사, i, me, my, it, this, that , is are 등 과 같은 단어는 빈번하게 등장하지만 실제 의미를 찾는데 큰 기여를 하지 않는다.
Stopwords는 “to” 또는 “the”와 같은 용어를 포함하므로 사전 처리 단계에서 제거하는 것이 좋다.
NLTK에는 153개의 영어 불용어가 미리 정의되어 있다. 17개의 언어에 대해 정의되어 있으며
한국어는 없다.

NLTK 데이터를 설치하고 그 데이터로 불용어를 제거할텐데요,
NLTK 데이터는 용량이 크기 때문에 따로 다운로드 받게 되어 있어요,
설치하는 다운로더가 약간 문제가 있어서 잘 설치가 되지를 않아요,
그래서 오늘 코드 비디오에 nltk데이터 설치하는 방법을 올려놓았는데요,
그 방법을 참고해서 설치해주시면 될 것 같구요,

words =[w for in words if not w in stopwords.words(‘english’) ]

스테밍(어간추출, 형태소 분석)
위키피디아에서 정의한 어간추출은 어형이 변형된 단어로부터 접사 등을 제거하고
그 단어의 어간을 분리해내는 것

Cats catlike catty 등 = cat
Stemmer, stemming stemmed 의 어간은 stem

message, messages, messaging과 같이
복수형, 진행형 등의 문자를 같은 의미의 단어로 다룰 수 있도록 도와줌.
Stemming(형태소 분석)
여기에서는 NLTK에서 제공하는 형태소 분석기를 사용.
포터 형태소 분석기는 좀더 보수적익고, 랭커스터 형태소 분석기는 좀 더 적극적임.
형태소 분석 규칙의 적극성 때문에 랭커스터 분석기는 더 많은 동음이의어 형태소를 생산한다.

여기에서는 스노우볼 스테머를 통해서 어간추출을 해보도록 할게요.
going->go
started->start
listening->listen.. 등

Lemmatization 음소표기법
lemmatization은 단어의 보조 정리 또는 사전 형식에 의해 식별되는 단일 항목으로 분석될 수
있도록 굴절된 형태의 단어를 그룹화하는 과정이다. 예를 들어 동음이의어가 문맥에 따라 다른의미를 갖는데
1)배가 맛있다.
배를 타는 것이 재미있다.
평소보다 두배로 많이 먹어서 배가 아프다.
위의 배는 모두 다른 의미를 갖는다.
영어에서 meet는 meeting으로 쓰였을때 회의를 뜻하지만 meet일 때는 만나다는 뜻을 갖는데
그 단어가 명사로 쓰였는지 동사로 쓰였는지에 따라 적합한 의미를 갖도록 추출하는 것이다.

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
‘fly’
‘flies’

for문 반복 학습-> 5000개 로그 찍기-> 단일코드에 %time 및 .apply로 트레인-> 멀티프로세싱

https://gist.github.com/yong27/7869662 참조.
또한 멀티프로세싱 하는 방법에 대해서는
http://www.racketracer.com/2016/07/06/pandas-in-parallel/ 
에 자세히 나와있음.

깃헙에 있는 노트북에서 참고해도됨.

멀티프로세싱으로 데이터 전처리를 하도록 했더니 
트레인 데이터와 테스트 데이터가 처리가 된 것을 확인할 수 있고,

워드 클라우드
-단어의 빈도 수 데이터를 가지고 있을때 이용할 수 있는 시각화 방법
-단순히 빈도 수를 표현하기 보다는 상관관계나 유사도 등으로 배치하는 게 더 의미있기 때문에 큰 정보를 얻기는 어렵다.

워드클라우드도 시간이 오래걸림.
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
노트북 안에서 보이도록 해당 설정을 해줘야지만 노트북에서 보인다.

Seaboard as sis
그리기
——————————
CountVectorizer로 텍스트 데이터 벡터화
이번 비디오에서는 이렇게 전처리한 데이터를 바탕으로 텍스트 데이터를 벡터화하는 작업을
해볼게요

텍스트 데이터를 기계가 이해할 수 있도록 하기위해서 벡터화 시켜줄거구요,

백 오브 워드라는 것을 다시 복습하고 가볼도록할게요.

문장에서 유니크한 단어들만 뽑아서 단어의 가방에 넣어주는 것,

[“John”, “likes”, . … ]
[1, 2, 1, 1, 2 ,1 ,1 ,0 ,0,0]
단어가방에서 이 문장 단어가 몇번 등장하는지
(각 토큰이 몇번 등장하는지 횟수를 세어준다.)
 n-gram을 사용해 bigram으로 담아주면
두개의 토큰씩 묶어서 단어가방에 담아주고,
John likes,
likes to
to watch,
watch movies
Mary likes
likes movies
…


데이터를 숫자화된 데이터로 바꿔주는 작업을 하게 될 것.

여기서는 사이킷런에 있는 countVectorizer를 통해 피처 생성
-정규표현식을 사용해 토큰을 추출한다.
-모두 소문자로 변환시키기 때문에 good, Good, gOod이 모두 같은 특성이 된다.
-의미없는 특성을 많이 생성하기 때문에 적어도 두개 문서에 나타난 토큰만을 사용한다.
-min_df로 토큰이 나타날 최소 문서 개수를 지정할 수 있다.

튜토리얼과 다르게 파라메터 값을 수정,
파라메터 값만 수정해도 캐글 스코어 차이가 많이 남

위의 countVectorizer를 fit 트랜스폼을 사용해서 학습할텐데 그냥 사용하면 속도가 오래걸리므로
파이프라인을 사용함.

파이프라인을 사용하면 그냥 fit transform하는 것보다 훨씬 빨리 끝남.

이렇게 학습된 데이터 피쳐가 어떻게 생겼는지 shape를 찍어보면
리뷰데이터가 2만5천개였기 때문에 
2만5천개의 행수와 2만개의 컬럼수를 갖게 되는 행렬데이터로 생성된 것을 볼 수 있음.

어떤 vocabulary를 가지고 있는지
Get_feature_names()로 찍어보면
‘aag’ , ‘aaron’, ‘ab’, ‘abandon’, ‘abbey’.. 등

벡터화 된 피처를 확인해봄

numpy 사용
배열로 보면 자세히 보기 어려워서
pandas의 데이터프레임을 사용하고, 컬럼에 vocab를 지정해서 봄
—————————————————
이전에서는 정제하고 벡터화하는 작업까지 했는데
이번에서는 벡터화된 데이터를 바탕으로 postivie인지 negative 인지 예측해보는 내용

랜덤포레스트의 개념을 먼저 알아보고 가도록 할게요
이 이미지가 scikit-learn을 가장 잘사용해서 가져왔구요,
scikit-learn에 있는 random forest를 가져와서 사용하구요,
Classification, regression, clustering, dimensionality reduction

분류, 클러스터링, 회귀, 차원 축소

회귀:
오늘코드비디오에서 자전거 대여량 예측했던 문제같은 경우는
몇대의 대여량이 있을 건지 시계열 데이터에서 예측하는 거라 regression에 해당함.

튜토리얼 3에 보면 비슷한 단어를 군집화해서
K-means 알고리즘을 사용하는데 이는 클러스터링에 해당

차원축소:
차원축소를 통해 워드투벡으로 벡터화된 t-SNE를 사용할텐데 이는 차원축소임.


supervised machine learning은 
unsupervised machine learing과의 가장 큰 차이점은
레이블 데이터가 있느냐 없느냐임

x_train. y_train
행렬 벡터

Decision Trees의 장점을 모아서 만든게 Random Forests임
투표로 결정
depth를 어떻게 결정해주냐에 따라 overfitting이나 언더피팅이 되기도함.

과소적합:
너무 간단한 모델이 선택되어 
= 학습이 제대로 되지 않은 상태, 너무 단순한 모델이 된 상태
 
과대적합:
너무 복잡한 모델을 만들어 일반화가 되지 않은 모델

랜덤포레스트의 위키피디아 정리:
분류, 회귀 분석등에 사용되는 앙상블 학습 방법의 일종으로, 훈련 과정에서 구성한 다수의 결정 트리부터 분류 또는 평균 예측치를 출력함으로써 동작함.

랜덤포레스트는 검출, 분류, 회귀 등 다양한 어플리케이션에 활용됨.


결정 트리를 이용한 방법의 경우, 결과 또는 성능의 변동 폭이 크다는 결점을 가지고 있음.
학습 데이터에 따라 생성되는 결정 트리가 다르기 때문에 , 그 방법을 보완한게
랜덤 포레스트임.

forest = RandomForestClassifier(
	n_estimators= 크게 지정할 수록 좀더 좋은 성능을 냄
	n_jobs = -1로 정의하면 모든 코어 사용, 2,3이면 두개나 세개의 코어를 사용하라고 지정.
	random_state=2018)
	랜덤포레스트를 여러번 돌릴때마다 다른 스테이트가 나오는데
	같은 스테이트가 나오도록 지정.

roc curve 로 크로스벨리데이션을 해서 스코어를 계산.

Pipeline 사용해서 여러개 스레드 만들어서 벡터화(테스트 데이터를 벡터화)

6번째 데이터의 100개의 데이터만 뽑아봄.
벡터화 된 단어로 숫자가 문서에서 등장하는 횟수를 나타낸다. (테스트데이터)

result = forest.predict(test_data_features)
 result =[:10]

예측 결과를 저장하기 위해 데이터 프레임에 담아준다.
quoting 3 csv파일을 불러왔을때와 동일한 형태로.


	
