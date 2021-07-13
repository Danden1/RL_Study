# rainbow

rainbow 는 기존의 dqn 시리즈들을 섞어놓았다. 그래서 사전 지식이 좀 많이 필요하다. 

## 사전지식

### Prioritized Experience Replay(PER)

기존의 dqn에서는 replay memory에서 무작위로 뽑았었다. 

PER은 중요한 경험을 자주 재사용하도록 하는 것이다.

TD error를 구하고, 이를 통해 prioritized buffer 를 만든다.

![1](./1.PNG)

k는 buffer에 들어있는 transition의 총 갯수이고, alpha는 0이면 uniform, 1이면 greedyg 이다. 얼마나 prioritization에 의한 sample을 할 것인지 나타낸다.



Prioritized Replay는  bias를 가져오는데, 주로 expectation에 대한 분포가 update마다 바뀌기 때문이다.

논문에서는 importance-sampling(IS) weights를 이용해 bias를 잡는다.

![2](./2.PNG)