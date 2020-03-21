## 00. 강화학습 개념정리먼저 해봅시다.

1. 강화학습을 푼다는 것은 **최적의 정책함수(=미래에 얻을수 있는 보상함수의 기댓값을 최대로 하는 행동)**를 찾는것과 같다.

2. 특정 환경과 상호작용하여, **보상을 최대화** 하는 행동 또는 행동 순서를 학습 

3. 미래에 얻어지는 보상도 다 고려해야한다. 

   

4. state  :  상태 , 선택하기 위해 받는 정보

5. reward :  agent 가 action을 하였을때 받는 보상

6. environment : 모든 환경

7. action : agent가 특정 환경에서 취할수 있는 행동

8. policy : 특정 state 에서 action 을 골라주는 함수 (어떤 액션을 골라줄것인가?)

9. episode : 시작 ~ 종료까지 agent가 거친 sequence

   

10. 강화학습에는 라벨이 없다. 시행착오를 겪으면서 학습 데이터를 스스로 모은다!

11. 강화학습의 데이터는 time series data : 각각의 데이터들이 독립적이지 않다는것. 상관관계를 가지고 있다.

12. 보상이 바로 정해지지 않는다. 여러 행동을 조합해서 reward를 받기도 한다. ( 그전에 내 행동이 맞는지 아닌지 알기 어렵다.)

13. action에 따라 다음 state는 불확실하다. ( 상대방이 어떻게 행동을 취할지 불확실하다. )

14. 현재의 결정, action이 이후에 올 데이터에 영향을 미친다.

    

15. **MDP : 마르코프 의사결정과정**

    - 현재가 주어졌을때, 과거와 미래가 독립적임
    - **t+1 번째의 state 는 오직 t번째  state 및 action 에만 의존함**
    - 현재의 나의 상태에는 과거의 상태가 "반영"은 되어있음.  그렇다고 미래를 결정하는것이 과거라는건 아님

16. 전이함수 : 특정 상태에서 특정 행동을 했을때, 다음번에 도달할 상태들의 확률을 나타낸것. (상대방이 어디에 둘것이냐 ? )

17. reward function : t+1번째 reward를 예측하는 함수

    

18. **Bellman equation : 벨만 방정식**

    - sequence s' 의 다음 time stamp 에서의 optimal Q-function 값이 모든 action a ' 에 대해서 알려져 있다면,  optimal strategy는  r+γQ∗(s′,a′)의 expected value를 maximize하는 것이라는 것이다.

19. MDP 가 주어졌을때, 최적의 정책 함수를 찾는 가장 기본적인 방법

20. Value function : 현재의 어떤 state에서 출발하여 얻을수 있는 모든 보상들의 합의 기댓값.

21. **할인률** : 같은 보상을 얻을수 있다면 빨리 얻는것이 좋음.

22. 가치반복 

    - 현재의 policy가 optimal하다고 전재하고, max를 취한다.
    - policy 가 optimal 하다는건 >> 

23. 정책반복 : 정책 평가, 정책 개선 (ㅠ)

    - policy에 따라 value function이 확률적으로 주어진다. 
    - 기댓값으로 value function을 구하게 된다. 





## 01. 논문리뷰 시작 

### Playing Atari with Deep Reinforcement Learning

#### 00. Abstract

- 강화학습을 통해 **successfully learn control policies directly from high demensionoal sensory input** 

-  Atari는 CNN 모델을 사용하며, 변형된 Q-learning을 사용하여 학습되었습니다. 

  - input : raw pixels 
  - output : value function estimating future rewards (미래의 보상을 예측하는 가치함수)

- 게임을 학습할때 픽셀 값들을 입력으로 받고, 각 행위에 대해 점수를 부여하고, 어떤 행동에 대한 결과값을 함수를 통해 받게된다.

-  Atari games중에서 7개의 게임환경을 통해 이 방법을 적용했고, 7개중 6개의 게임에서 이전 모든 접근들을 능가했고, 6개 중에서 3개는 인간을 넘어섰다.

  

#### 01. Introduction

- **vision , speech** 같은  **high demensionoal sensory input** 으로 부터 agent를 학습시키는것은 강화학습의 오랜 과제였다.

-  딥러닝이 발전함에 따라 Vision, Speech와 같은 고차원의 데이터들을 추출하는 것이 가능해졌다.

- 하지만 딥러닝을 강화학습에 적용하는 과정에서 몇가지 문제점을 발견하게 되었다,

  - deep learning :  label 된 많은 양의 training 데이터를 필요로 함.

  - reinforce learning : sparse, noisy, **delayed (between action and rewards)****한 reward signal 라는 scalar 값을 통해 학습한다.

  - > RL에서는 어떠한 행위를 하면 그 행위에 대한 결과를 알기까지 시간이 필요하다. 그리고 이런 delay는 어려움을 유발 한다는것이죠~!~!

    

  - **:star:딥러닝 알고리즘에서 각 데이터 들은 독립적이지만, 강화학습에서는 하나의 행위가 다른것들과 연관성이 높다.**

  

  - RL 에서는 알고리즘이 새로운 행동을 배울때마다 data 의 distribution이 변하게 되는데, 이것은 데이터의 분포가 고정되어있다고 가정하는 딥러닝의 가정과 충돌하여 문제가 될수 있다. 



- 그렇지만, 이 논문에서는 그것을 극복한, CNN이 복잡한 RL 환경에서 성공적인 학습을 할수 있음을 증명한다.

-  :star:변형된 **Q-Learning을** 통해 학습되며, weight를 update하기 위해 **stochastic gradient descent**를 사용합니다.

- correlated data문제를 해결하기 위해 **experience replay mechanism**을 사용한다. 

  - 이전에 학습한 데이터들을 저장하여 random한 샘플로 사용하고, 그렇게 함으로써 training distribution이 smooth 하도록 만드는 기법.

    

- 우리의 목표는 **to create single neural network agent that is able to sucessfully learn to paly as many of the games as possible.** (가능한 많은 게임들을 성공적으로 학습할수 있는 NN을 만드는것이다.)

- **:star:NN은 어떠한 게임에 대한 특정 정보나 게임의 우위를 위한 데이터등을 제공받지 않는다.**. 

- **:star:오직 비디오의 시각 데이터와 Reward 그리고 터미널로부터 오는 신호 그리고 가능한 몇개의 행동으로만 학습을 진행하였습니다.**

- 또한 다양한 게임들에 동일한 network architecture과 hyper parameter를 사용했다.



#### 02. Background

-  Reinforcement Learning을 위해서는 먼저 **환경 E**을 정의해야한다. 이 논문에서는 **Atari 에뮬레이터**가 환경이 될 것이다. 
- 매 시간마다 agent는  legal game action A={1,…,K} 중에서 action 을 하나 선택한다.
- 게임을 하는동안 컨트롤러의 버튼을 누르는 action이 모이게 되면 현재 점수에 어떻게든 영향을 주게되고, 그 결과로 최종 score가 결정된다. (action 이 모여서 score 결정)
-  즉, Atari 게임 환경에서 reward 는 게임 score이다.
- 현재 내가 선택한 action이 바로 reward 에 반영되는것이 아니라 나중에 반영될수도 있음. (received after many thousands of time-steps have elapsed.)



- 이 논문에서는 **vision 데이터를 사용해서 state를 정의**하는데,  간단하게 xt를 state로 삼으면 될 것 같지만 그렇지 않다.

- :star:실제로는 화면 하나만 보고 알 수 있는 정보가 제한적이고 **현재 상태를 정확하게 판단하기 위해서**는 vision정보와 내가 행한 action을 포함한 **과거 history**들까지 모두 있지 않으면 안되기 때문에 이 논문은 **state 를 action과 image의 sequence로 정의한다.** :star:

  - :star:이 사실때문에 state를 image와  action 의 sequence로 정의 한다.

    

- 그리고 게임은 언젠가 끝나기 때문에 finite 한 MDP가 된다. 

- 목표 : 가장 높은 점수를 획득하는것.  ( ! 시간이 오래 지날수록 해당 reward의 가치는 내려간다. = 할인률 )



- Q∗(s,a)=maxπE[Rt|st=s,at=α,π]   : 정책 π 를 통해 얻을수 있는 reward의 maximum expected value

  - 최적의 Q fuction은 bellman equation이라는 특성을 따른다. 

  -  MDP 에서는 이 **optimal action value function 혹은  optimal Q-function 하나만 제대로 알고 있다면**,  반드시 **항상 optimal한 action을 고를수 있다.** 

  - 이때 optimal strategy 는 r+γQ∗(s′,a′) 를 maximize 하는것

  - Q function : 각 state에서 어떤 행동 a를 했을때, 기대되는 미래 보상의 총합

    

-   RL에서는 모든 state와 action에 대한 labeled data가 없기 때문에 이를 어떻게 다뤄야 하는지를 모델에서 고려해야만한다. 또한 현재까지 연구된 많은 deep learning structure들은 data가 i.i.d.하다고 가정하지만, **실제 RL 환경에서는 state들이 엄청나게 correlated되어있기 때문에 제대로 된 learning이 어렵다.**  

  

- 이 논문에서 문제를 해결하기 위해 사용한 방법

  - 1. **Freeze taget Q-network:**

       optimization 과정에서 parameter θ가 update되는 동안 loss function Li(θi) 의 이전 iteration paramter θi−1은 고정된다는 것이다. 

       이렇게하는 이유는 supervised learning과는 다르게, target의 값이 θ의 값에 (민감하게) 영향을 받기 때문에 stable한 learning을 위하여 θ값을 고정하는 것이다.  

       

  - 2. **Experience replay :**

       agent 의 experience 를 각 time stamp 마다 다음과 같은 튜플 형태의 **메모리에 저장한후 이를 다시 이용하는것이다.**

        Experience replay를 사용함으로써 data의 correlation을 깰 수 있고, 조금 더 i.i.d.한 세팅으로 network를 train할 수 있게 된다. 또한 방대한 과거 데이터가 한 번만 update되고 버려지는 비효율적 접근이 대신에, **지속적으로 추후 update에도 영향을 줄 수 있도록** 접근하기 때문에 데이터 사용도 훨씬 효율적이라는 장점이 있다. 

       

  - 3.  **Clip reward or normalize network adaptively to sensible range** 

        reward의 값을 [-1,0,1] 중에서 하나만 선택하도록 강제하는 아이디어이다.  

         내가 100점을 얻거나 10000점을 얻거나 항상 reward는 +1 이다. ‘highest’ score를 얻는 것은 불가능하지만, 이렇게 설정함으로써 조금 더 stable한 update가 가능해진다.  



### 03. Deep Reinforcement Learning

- ![1584808125630](C:\Users\MiSung\AppData\Roaming\Typora\typora-user-images\1584808125630.png) 
- ![1584808238939](C:\Users\MiSung\AppData\Roaming\Typora\typora-user-images\1584808238939.png)
- 







### 참고자료

 https://mangkyu.tistory.com/60 

 http://hugrypiggykim.com/2019/03/10/playing-atari-with-deep-reinforcement-learning/ 

 http://sanghyukchun.github.io/90/ 