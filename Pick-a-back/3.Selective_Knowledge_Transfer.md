# Pick a back

# 3 Selective Knowledge Transfer

❤️ 저자의 목표: device-to-device knowledge federation을 고려하여 개별적인 continual learning을 분산 방식으로 향상시키고자 함 
<br>

❤️ 목표 해결을 위한 방법 제시: **Pick-a-back을 제안함**
    
### Pick-a-back이란?

비슷한 작업을 처리하는 인접 장치를 찾아 그 지식을 선택적으로 모델에 주입하여 연속 학습을 통해 적용하는 간단하면서도 효과적인 지식 연합 방식이다.

- 장치 간 협력 학습을 가능하게 한다.
- distributed continual learning의 문제를 해결한다.
- 각 장치에서 이전에 획득한 지식을 재사용할 수 있다.

⇒ 디바이스 간의 이질적인 데이터에서 효율적인 지식 선택, 공유 및 확장을 가능하게 하여, 작업이나 학습 시나리오 전반에 걸쳐 지속적인 학습 향상과 적응을 이끌어낸다.
    
<br>

## 3.1 Problem Formulation

❤️ 각 디바이스의 continual learning에 대한 설명

각 디바이스는 새로운 작업을 학습할 때, 이전에 학습한 모델을 기반으로 학습하는 continaul learning방식을 활용한다.

- continual learning의 효과: 이를 통해 각 디바이스는 이전 작업에서 얻은 지식을 기반으로 새로운 작업을 학습하면서도, 기존의 지식을 유지할 수 있다.

- continaul learning의 목표: 각 디바이스가 자신의 연속 작업에 대한 모델을 최적화하고, 모델의 정확도를 최대화 할 수 잇는 reference model을 찾는 것이다.

<br>

## 3.2 Architecture

❤️ pick-a-back의 학습 구조(continual learner와 federated learner)

### 1. continual learner for local training
    
task-incremental learning(여러 작업들을 순차적으로 처리함)상황을 가정한다. 

- 각 디바이스는 자신의 작업(task)를 수행하기 위해 continual learning을 사용한다.
    1. catastrophic forgetting을 해결할 수 있다.
    2. 새로운 모델을 이전에 학습한 지식 위에 연결하므로 **메모리 사용을 줄이고 모델의 신뢰도**를 높일 수 있으며, 특히 **엣지 AI** 환경에서 매우 가치 있음
- 논문에서는continaul learning의 방식으로 CPG라는 방법을 채택한다.
    
    CPG는 필요할 때 모델을 확장, 압축하고 비효율적인 뉴런을 마스킹하여 네트워크를 재사용 할 수 있는데 연속 학습 방법
    
    https://ffighting.net/deep-learning-paper-review/incremental-learning/cpg/

<br>
    
### 2. federated learner for knowledge transfer
    
각 디바이스는 프라이버시 및 통신 오버헤드의 문제로 개인의 데이터를  공유하지 않는다. 따라서 데이터 공유 대신 **모델 공유 접근 방식**을 채택한다.

- 디바이스가 다른 디바이스의 지식에 노출됨으로써, 모델은 향상된 특징 표현이 가능하며 특정 학습 목표에 유리한 더 넓은 패턴과 정보를 포착할 수 있게 해주는 장점이 있음
- 지식을 가져올 외부 디바이스를 선택하기 위해서  **decision pattern similaritiy**를 사용한다.
    
    decision pattern이란? 모델이 어떤 데이터를 어떻게 처리하고 분류하는지를 나타내는 학습 방식으로, 모델이 데이터에서 추출하는 특징과 이 특징들을 바탕으로 하는 분류 방식이 해당한다.
        

**정리**: 디바이스는 continual learner 모드와 federated learner 모드를 번갈아가며 작업을 수행한다. 이렇게 획득한 지식을 기존의 자신의 지식과 결합하여 학습자는 잊지 않고 누적된 지식을 구축할 수 있다.

<br>

## 3.3 Selective Knowledge Federation

❤️ 선택적 지식 전이 개요

각 디바이스는 서로 다른 작업(task)을 학습 하며, 각 작업은 고유한 positive and negative knowledge가 존재한다. <br>
(positive knowledge: 학습에 유용한 지식 / negative knowledge: 학습에 불리한 지식)

이때 외부 디바이스 중 본인과 비슷한 작업을 수행하며 이와 관련된 전문적인 지식을 학습했을 수도 있다. 따라서 선택적인 지식 전이를 사용하면 각 모델의 성능을 향상시키는 효과를 기대할 수 있다.

<br>

**⇒ 그렇다면 외부 디바이스를 선택하는 기준은?**

여러 작업을 수행하는데 있어 유사한 패턴을 가진 경우 학습자에게 더 일반화된 지식을 제공할 가능성이 높으며 더 넓은 통찰력과 새로운 지식을 통해 학습 속도를 향상할 수 있을 것이므로! Decision pattern similarity를 사용함

하지만, 모델의 유사성이 유사한 지식을 의미하지 않으므로, 모델 자체보다는 **결정 패턴의 유사성**에 포인트를 맞춤

<br>

❤️ 외부 디바이스를 선택하는 방법

=> Decision pattern의 유사성을 비교한다!

**ModelDiff**

입력 데이터에 대한 반응 비교하여 decision pattern의 유사성을 계산한다. 

- 계산 방법
    1. request model에서 입력 데이터와 적대적 데이터를 샘플링한다. 
    2. 그 후, 데이터를 request model f와 외부 모델 g에 입력하고 두 모델이 내리는 결정을 DDV(Decision Distance Vector - 두 모델의 출력 사이의 거리)로 계산한다. 
        - 외부 모델은 가장 최근 업데이트가 완료된, 즉 request model보다 한 단계 전의 모델이어야 한다.
            why? 
            - 계산 효율성 : 연합 학습에서 모든 디바이스가 실시간으로 모델을 동기화하고 비교하는건 매우 큰 계산 비용이 소모됨
            - 동기화 문제 해결
            - 신뢰성: 이전 단계의 모델은 이미 과거 데이터를 바탕으로 충분히 학습된 모델이기 때문에 신뢰할 수 있는 지식으로 간주됨
    3. 마지막으로, 두 모델의 DDV사이의 cosine similarity를 비교하여 가장 유사한 결정 패턴을 가진 모델을 선택한다.
 
<br>

❤️ 외부 디바이스와의 지식 전이

1. 선택된 외부 디바이스의 모델을 가져와 백본으로 사용하여 request model의 현재 작업을 재학습한다. 
    
    (이 과정에서 외부 모델 전체를 통째로 사용하는 것이 아닌, 결정 패턴이 유사한 부분만 로컬 작업에 활용하는 것임 ⇒ Selective!)
    
2. 외부 모델을 활용하여 최종 로컬 모델이 구축된다.
