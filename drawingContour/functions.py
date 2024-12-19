#### 외곽선 시각화 함수  (just line)
def draw_segmentation_with_contours(image, mask, color=(255, 0, 0), thickness=2):
    """
    이미지에 마스크 외곽선을 추가하는 함수.
    """
    image = image.copy()
    mask = mask.astype(np.uint8)  # 마스크를 이진화 (0과 1로만 구성)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)  # 외곽선 그리기
    return image



#### 네온 외곽 함수
def draw_neon_contours(image, mask, neon_color=(255, 0, 255), thickness=3, blur_size=21, glow_strength=3):
    """
    네온 효과를 외곽선에 적용하는 함수.
    Parameters:
        image (numpy.ndarray): 입력 이미지
        mask (numpy.ndarray): 이진화된 마스크 (0과 1로만 구성)
        neon_color (tuple): 네온 색상 (B, G, R)
        thickness (int): 외곽선 두께
        blur_size (int): 흐림 효과의 강도
        glow_strength (int): 흐림을 여러 번 반복해 네온 효과 강화
    """
    output = np.zeros_like(image)
    mask = mask.astype(np.uint8)

    # 외곽선 그리기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, neon_color, thickness, lineType=cv2.LINE_AA)

    # 네온 빛 흐림 효과 반복 적용
    neon_glow = np.zeros_like(output)
    for _ in range(glow_strength):
        output = cv2.GaussianBlur(output, (blur_size, blur_size), 0)
        neon_glow = cv2.addWeighted(neon_glow, 1.0, output, 0.5, 0)

    # 최종 네온 효과와 이미지 합성
    result = cv2.addWeighted(image, 1.0, neon_glow, 0.6, 0)
    cv2.drawContours(result, contours, -1, neon_color, thickness, lineType=cv2.LINE_AA)

    return result
