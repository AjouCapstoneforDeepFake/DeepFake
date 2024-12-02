# DeepFake

# **FakeME: 한국인 기반의 Deepfake 생성 및 탐지 시스템**

**FakeME**는 **DeepFake**의 **Fake**와 **Media**의 **ME**를 결합하여 만들어진 이름으로, Deepfake 기술을 활용한 생성 및 탐지 기능을 통합적으로 제공하는 시스템입니다.

---------

## **프로젝트 개요**

**Deepfake**란 DeepLearning과 Fake의 합성어로, 딥러닝 기술을 사용하여 인간 이미지를 합성하는 기술입니다.  
최근 몇 년간 Deepfake의 핵심 기술인 **생성적 적대 신경망(GAN)** 기술의 발전으로, Deepfake 콘텐츠는 육안으로 구분하기 어려울 정도로 높은 수준에 도달하였습니다.

그러나 이 기술은 다음과 같은 **양면성**을 가지고 있습니다:
1. **부정적 사례**  
   - 디지털 성범죄 사건 (예: N번방 사건)  
   - 허위 정보 확산 사례 (예: 대통령, 사업가 관련 가짜 뉴스)  
   - 대한민국 피해자의 비율: 전 세계 Deepfake 피해자의 53% 차지  
2. **긍정적 사례**  
   - 교육: 독립운동가의 모습을 재현하여 실감나는 교육 자료 제공으로 학습 효과 증대  
   - 의료: 고인의 모습을 재현하여 유가족 위로  
   - 광고 및 영화: 비용 절감 및 콘텐츠 몰입도 증대  

FakeME는 이러한 Deepfake 기술의 **양면성**을 연구하여, **생성과 탐지 기능을 동시에 처리하는 통합 시스템**을 구축함으로써 기술의 긍정적 잠재력을 극대화하고 부정적 영향을 최소화하기 위해 개발되었습니다.

---

##  주요 기능

### 기존 Deepfake 서비스의 한계
1. **기능의 단일화**  
   - 생성 또는 탐지 중 하나의 기능에만 집중
2. **한국인 데이터를 활용한 Deepfake 생성 및 탐지 기술이 부족**  
   - 글로벌 데이터셋(FaceForensics++ 등) 기반 모델로 한국인 얼굴 특징에 최적화되지 않음
3. **Deepfake 악용 방지 시스템 부재**  
   - SNS 및 동영상 공유 플랫폼에서 콘텐츠 확산 차단 불가

### 위와 같은 한계를 극복하기 위해 다음과 같은 요구사항 분석을 통해 시스템 개발을 진행
1. **통합 시스템 제공**  
   - Deepfake **생성 및 탐지 기능을 통합적으로 제공**
2. **한국인 데이터 최적화**  
   - 한국인 이미지 데이터를 기반으로 학습한 모델로 **한국인 얼굴 특징에 최적화**
3. **Deepfake 악용 방지 시스템**  
   - SNS 및 동영상 공유 플랫폼에서 Deepfake 콘텐츠가 가장 빠르게 확산된다는 점을 고려하여 플랫폼 내에 본 시스템을 도입하여 운영
   - Deepfake 게시물임을 밝혀 **부적절한 콘텐츠 확산을 사전에 방지**하고 **사람들이 오해하여 발생할 수 있는 허위 정보 확산을 방지**

---

## **서비스 및 시스템 구조**

1. **SNS 미 동영상 공유 플랫폼 내에서 본 시스템을 도입하여 운영**
2. **한국인 이미지 데이터를 기반으로 학습하여 한국인의 얼굴 특징에 최적화된 모델 개발
3. **Deepfake 생성 및 탐지 기능을 통합적으로 제공하는 시스템 개발

   


---

---
