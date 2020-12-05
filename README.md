# deep-learning-zero-to-all-study

## Lec 1
1. supervised learning이란?
- training set(정해져있는 또는 labeled 된 데이터)을 갖고 학습하는 것
2. unsupervised learning이란?
- data(unlabeled data)를 보고 스스로 학습하는 것
3. training data set이란?
- 모델을 학습하기 위한 dataset?
4. supervised learning types 3가지?
- regression, binary classification, multi-label classification
5. data flow graph-node
- mathematical operation
6. data flow graph-edge
- multi-dimensional data array(tensor)
7. rank, shape, type
- rank: array의 차원
- shape: element에 몇개가 들어있는지
- type: data type

## Lec 2
1. Linear Regression이란? 
- linea한 model이 우리가 가진 데이터와 맞을 것이다 라는 가설을 세우고 분석하는 것 
- H(x) = Wx + b
2. Linear Regression의 목표
- 가장 작은 값을 갖는 cost(W,b)를 구하는 것
3. cost/lost function이란
- 가설과 실제 data의 거리 계산 
- ![eq1](./img/eq1.png)
  - 여기서 m은 데이터의 갯수

## Lec 3
1. gradient descent algorithm이란?
- 경사를 따라 내려가는 알고리즘
2. gradient descent algorithm의 동작 원리
- initial guesses에서 parameter(W,B)를 조금씩 바꿔서 local minimum을 찾는 것
3. 사용되는 공식
- ![eq1](./img/eq2.png)
  - 여기서 α는 learning rate 
4. cost function을 설계할 때 주의할점 
- cost function의 모양이 convex function이 되는지 확인필요