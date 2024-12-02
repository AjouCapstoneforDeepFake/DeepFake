import cv2
import numpy as np
from skimage import transform as trans

# 얼굴 랜드마크의 좌표 템플릿 정의 (5개의 주요 얼굴 점)
# src1 ~ src5는 다양한 각도의 얼굴 랜드마크 좌표
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]], dtype=np.float32)

src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]], dtype=np.float32)

src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]], dtype=np.float32)

src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]], dtype=np.float32)

src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]], dtype=np.float32)

# 랜드마크 템플릿을 하나의 배열로 묶음
src = np.array([src1, src2, src3, src4, src5])
src_map = src

# FFHQ 얼굴 정렬 템플릿 (512x512 해상도 기준)
ffhq_src = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]])
ffhq_src = np.expand_dims(ffhq_src, axis=0)

# 얼굴 랜드마크 정렬 행렬 계산
def estimate_norm(lmk, image_size=112, mode='ffhq'):
    """
    얼굴 랜드마크를 기반으로 정렬 행렬 계산
    lmk: 입력 랜드마크 (5개의 주요 점)
    image_size: 출력 이미지 크기
    mode: 'ffhq' 또는 다른 모드에 따라 템플릿 사용
    """
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)  # 랜드마크에 1 추가 (동차 좌표)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'ffhq':
        src = ffhq_src * image_size / 512  # FFHQ 템플릿 비율 조정
    else:
        src = src_map * image_size / 112  # 다른 템플릿 비율 조정
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])  # 정렬 행렬 추정
        M = tform.params[0:2, :]  # 2x3 행렬 추출
        results = np.dot(M, lmk_tran.T).T  # 랜드마크 변환
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))  # 변환 오차 계산
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index  # 최적의 정렬 행렬 및 인덱스 반환

# 정렬된 얼굴 이미지 생성
def norm_crop(img, landmark, image_size=112, mode='ffhq'):
    """
    입력 이미지를 정렬하고 크롭
    img: 입력 이미지
    landmark: 랜드마크 좌표
    image_size: 출력 이미지 크기
    mode: 'ffhq' 또는 'newarc' 모드
    """
    if mode == 'Both':
        M_None, _ = estimate_norm(landmark, image_size, mode='newarc')
        M_ffhq, _ = estimate_norm(landmark, image_size, mode='ffhq')
        warped_None = cv2.warpAffine(img, M_None, (image_size, image_size), borderValue=0.0)
        warped_ffhq = cv2.warpAffine(img, M_ffhq, (image_size, image_size), borderValue=0.0)
        return warped_ffhq, warped_None
    else:
        M, pose_index = estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

# 정사각형 크기로 이미지 크롭
def square_crop(im, S):
    """
    정사각형 크기로 이미지 크롭 및 리사이즈
    im: 입력 이미지
    S: 출력 크기
    """
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))  # 크기 조정
    det_im = np.zeros((S, S, 3), dtype=np.uint8)  # 빈 정사각형 이미지 생성
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale

# 변환 행렬을 사용하여 이미지 변환
def transform(data, center, output_size, scale, rotation):
    """
    이미지 변환 (회전, 크기 조정, 중앙 정렬)
    data: 입력 데이터
    center: 중심 좌표
    output_size: 출력 크기
    scale: 크기 비율
    rotation: 회전 각도
    """
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0  # 라디안으로 변환
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4  # 모든 변환을 결합
    M = t.params[0:2]  # 2x3 변환 행렬
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M

# 2D 점 변환
def trans_points2d(pts, M):
    """
    2D 좌표 변환
    pts: 입력 좌표
    M: 변환 행렬
    """
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)  # 변환 적용
        new_pts[i] = new_pt[0:2]
    return new_pts

# 3D 점 변환
def trans_points3d(pts, M):
    """
    3D 좌표 변환
    pts: 입력 좌표
    M: 변환 행렬
    """
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
    return new_pts

# 좌표 변환 함수
def trans_points(pts, M):
    """
    좌표 변환 (2D 또는 3D)
    pts: 입력 좌표
    M: 변환 행렬
    """
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
