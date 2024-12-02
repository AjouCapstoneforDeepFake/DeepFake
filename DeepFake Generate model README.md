# DeepFake Model Overview

# Key Concepts

### Source Face
- 교환할 원래 얼굴

### Target Face
- 교체될 얼굴이 적용될 대상 얼굴

### Identity and Attribute
- **Identity**: 눈, 코, 입, 피부 톤과 같은 독특한 얼굴 특징이 포함
- **Attribute**: 위치, 표현, 각도와 같은 컨텍스트 기반 또는 감정적 특징이 포함
- **Goal**: 대상 얼굴의 속성을 적용하면서 원본 얼굴의 동일성을 유지

## 구현 방법

### 핵심 접근 방법
- 얼굴의 고유한 **identity**를 유지하도록 Encoder와 Decoder 사이에 추가적인 **identity-preserving model**을 삽입.

---

## 모델 구조

### Encoder
- **Source image**에서 **feature**를 추출.
- **Target image**의 feature를 **Source image**의 identity에 맞게 변환하되, **attribute 정보**는 변경하지 않음.
- **Training loss**를 사용하여 네트워크가 최적화되도록 개선.

#### Encoder 주요 단계:
1. **Identity 추출**: Source image에서 identity와 attribute 정보를 추출.
2. **Identity 벡터 생성**: Source image의 identity를 기반으로 identity vector 생성.
3. **Embedding**: Target image에 Source image의 identity를 주입하여 원래 identity를 대체.

---

### Decoder
- Identity 정보가 주입된 수정된 feature를 decoder에 통과.
- 수정된 feature로부터 이미지를 생성.

---

## Model Pipeline

1. **Feature Extraction**:
   - **Source face**(Identity) 및 **Traget face**(attribute)에서 특징을 추출합니다.

2. **Identity Injection**:
   - 소스 얼굴의 identity를 대상 얼굴의 특징 공간에 삽입.

3. **Image Reconstruction**:
   - decoder를 사용하여 반영되는 최종 이미지를 생성
     - The **identity** of the source face.
     - The **attributes** of the target face.

---

## 코드 예제 (Pseudo-code)
```python
# Encoder
source_features = extract_features(source_face)  # Extract identity from source
target_features = extract_features(target_face)  # Extract attributes from target

# Combine Identity and Attribute
identity_vector = generate_identity_vector(source_features)
modified_features = inject_identity(target_features, identity_vector)

# Decoder
output_image = decoder(modified_features)  # Generate final image

# Training Loss
loss = compute_training_loss(output_image, source_face, target_face)
update_model(loss)
