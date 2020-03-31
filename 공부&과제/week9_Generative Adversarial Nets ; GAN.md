# <u>G</u>enerative <u>A</u>dversarial <u>N</u>ets ; GAN

### :zero: Abstract

- :point_right: G : generative model

  - 우리가 가지고 있는 data  x의 distribution을 알아내려고 한다. ( 데이터의 분포를 학습하는 모델 )

- :point_right: D : discriminative model 

  - sample 이 진짜인지 G가 만들어낸건지 구분하려한다. 
  - 각각의 경우의 확률을 estimate 함.

  ![image](https://user-images.githubusercontent.com/28949182/77851328-8b958b00-7213-11ea-88b2-269f03a56208.png) 

  

- G는 D가 실수할 확률을 높이게 하고 싶어하고(**max**) , D는 실수할 확률을 낮추고 싶어함(**min**)  

  - **  **"minimax two-player game or minimax problem"** **
  - **확률이 0.5 가 된다는 것은 서로 구분하지 못하는 것**. 즉, 이렇게 되는것이 목표다.



-  GAN은 이미지를 만들어내는 네트워크(**Generator**)와 이렇게 만들어진 이미지를 평가하는 네트워크(**Discriminator**)가 있어서 서로 대립(**Adversarial**)하며 서로의 성능을 점차 개선할 수 있는 구조로 만들어져있습니다. 



- G와 D는 multi layer perceptrons 으로 정의되고,  back propagation으로 학습된다.







### :one: Introduction

- D 는 sample 이  G model 로 부터 왔는지, 데이터로부터 왔는지 판별하는것을 학습한다.
- G 는 위조지폐를 만드는 쪽, D 는 위조지폐를 감지하는 경찰과 유사하다.
- 이 경쟁은 위조지폐가 진짜와 구별되지 않을때 까지 계속된다.



- adversarial nets : 두 네트워크가 서로 적대적인 관계다.

- 우리의 목적은 Generator 는 점점 더 실제 이미지와 닮은 샘플을 생성하게 만들고, Discriminator 는 샘플을 점점 더 잘 구별하는 모델을 만드는것이다.

  

- ![image](https://user-images.githubusercontent.com/28949182/77852181-2db77200-7218-11ea-9f0e-3feab2f8ff5d.png)

- 경찰에 입장에서 **x는 진짜 돈이니까 D(x) = 1 이 되고, 가짜돈인 z는 D(G(z)) = 0 이 되도록** 노력한다. 

  



### :two: Adversarial nets

- ![1585493339633](C:\Users\MiSung\AppData\Roaming\Typora\typora-user-images\1585493339633.png) 

  ​																	real data를 넣었을때 + fake data 를 넣었을때.



- 극단적인 예시를 생각해보자! 

  1. G가 진짜 이미지와 완벽히 닮은 샘플을 만든경우
     
     - V(G,D) 의 최솟값 = - inf
     
     - D가 이미지가 진짜일 확률이 1 이라고 잘못 결론 내리면, D(G(z)) = 1 
     - 두번째 항의 값이 - inf 
     
  2.  D가 완벽하게 구분할 경우

     - V(D,G) 의 최댓값 = 0

     - D(G(z))=0, D(x) = 1 이므로 
     - log 1 =0 



### :three:Theoretical result

- ![image](https://user-images.githubusercontent.com/28949182/77998555-38831b80-736c-11ea-9fd7-30646e2f75ed.png)
- GAN 은 파란 점선을 동시에 업데이트 하면서 학습이 된다. 
- 검은 점선 : data generating distribution  (원래 데이터 분포)
- 초록선 : 검은 점선으로 부터 비롯된 sample 을 generative distribution (generate된거 )
- Generative adversarial nets are trained by simultaneously updating the discriminative distribution
  (D, blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) px from those of the generative distribution pg (G) (green, solid line).



- G와 D가 점점 구분하기 어려워 지면 d 처럼 변한다.  = D가 확률을 0.5로 계산하게 된다. 



- 처음에 학습을 시작할때 G는 잘 generate 못하겠지?
  - 그니까 D 는 구분을 잘해낼거고,  D(G(z)) = 0
  - 이런경우에 log(1-D(G(z))) 는 최대값인 0 을 가지게 된다. 
  - 즉, log(1-D(G(z))) 를  최소화 해서 G를 학습하는것 보다.
  - log(D(G(z))) 를 최대가 되게 G를 학습하는것이 좋다.  
  - 이렇게 하면 학습 초기에 더 강한 gradient 를 받게 된다. = 학습이 더 잘 이루어 지게 한다. 



- ![image](https://user-images.githubusercontent.com/28949182/78002972-f1e4ef80-7372-11ea-93b5-8eb790440e01.png) 









### :four:Advantage & Disadvantage

- 장점
  - 확률 모델을 명확히 정의하지 않아도 Generator  자체가 만드는 분포로 sample을 생성가능하다.
  - 특정 모델을 가정해서 만들필요가 없다.
- 단점
  - 학습을 언제 정지 시킬지에 대한 명확한 기준이 없다.
  - minmax문제 이기 때문에 학습시키기가 어렵다.
  - 다른 모델과 비교 했을때 목적함수로 결과를 평가하기 어렵다.
  - discrete data를 생성하는것을 학습시키기 어렵다. 
  - 





### :five: Conclusion

- This framework admits many straightforward extension.
- ![image](https://user-images.githubusercontent.com/28949182/77999958-7719d580-736e-11ea-860c-851739fa7faa.png)
- 



### :six: more : 

- GAN 의 기본적인 구조는 맨 왼쪽과 같고, 다양한 구조들이 더 생겨났다고 한다.



- ![image](https://user-images.githubusercontent.com/28949182/77999576-da573800-736d-11ea-8af3-b6bfc0de25d2.png) 

