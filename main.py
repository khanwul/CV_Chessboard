import numpy as np
import cv2 as cv

# 설정값
input_video_file = './chessboard.mp4' # 입력 비디오 경로
output_video_file = './output.mp4'  # 출력 비디오 경로
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# 비디오 열기
video = cv.VideoCapture(input_video_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_video_file

# 카메라 켈리브레이션
print("카메라 캘리브레이션을 위해 프레임을 수집합니다...")

# 3D 세계 좌표계의 체스보드 포인트 생성
objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
objp *= board_cellsize

objpoints = [] # 3D points
imgpoints = [] # 2D points

frame_count = 0
while True:
    valid, img = video.read()
    if not valid:
        break

    # 10프레임마다 하나씩 코너 탐색
    if frame_count % 10 == 0:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        success, corners = cv.findChessboardCorners(gray, board_pattern, board_criteria)
        if success:
            objpoints.append(objp)
            # 코너 위치 정밀화 (서브픽셀 단위)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    frame_count += 1

assert len(imgpoints) > 0, "체스보드를 찾을 수 없습니다. 영상 경로를 확인해주세요."

print("카메라 매트릭스와 왜곡 계수를 계산 중입니다...")
ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("==== 계산된 파라미터 ====")
print("K:\n", K)
print("Distortion Coefficients:\n", dist_coeff)


# 비디오 생성 및 계단 모양 블록 그리기
print("\n새로운 영상을 생성합니다...")

# 비디오를 처음으로 되감기
video.set(cv.CAP_PROP_POS_FRAMES, 0) 

# VideoWriter 초기화
fps = video.get(cv.CAP_PROP_FPS)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v') # mp4 포맷 코덱
out_video = cv.VideoWriter(output_video_file, fourcc, fps, (width, height))

def draw_box(image, rvec, tvec, K, dist, x0, x1, y0, y1, z0, z1):
    # 지정된 3D 좌표 구간에 블럭을 그리는 헬퍼 함수
    lower = board_cellsize * np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    upper = board_cellsize * np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])

    line_lower, _ = cv.projectPoints(lower, rvec, tvec, K, dist)
    line_upper, _ = cv.projectPoints(upper, rvec, tvec, K, dist)

    cv.polylines(image, [np.int32(line_lower)], True, (255, 0, 0), 2)
    cv.polylines(image, [np.int32(line_upper)], True, (0, 0, 255), 2)
    for b, t in zip(line_lower, line_upper):
        cv.line(image, tuple(np.int32(b.flatten())), tuple(np.int32(t.flatten())), (0, 255, 0), 2)

while True:
    valid, img = video.read()
    if not valid:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    success, img_points = cv.findChessboardCorners(gray, board_pattern, board_criteria)
    
    if success:
        # 카메라 포즈 추정
        ret, rvec, tvec = cv.solvePnP(objp, img_points, K, dist_coeff)

        # 계단 모양 블록 그리기 (아래 2블록, 위 1블록)
        # 기준 위치 설정: 체스보드 중앙 부근 (x=4, y=3)
        # 1. 아랫단 왼쪽 블록
        draw_box(img, rvec, tvec, K, dist_coeff, x0=4, x1=5, y0=3, y1=4, z0=0, z1=-1)
        # 2. 아랫단 오른쪽 블록
        draw_box(img, rvec, tvec, K, dist_coeff, x0=5, x1=6, y0=3, y1=4, z0=0, z1=-1)
        # 3. 윗단 블록 (아랫단 왼쪽 블록 위에 위치)
        draw_box(img, rvec, tvec, K, dist_coeff, x0=4, x1=5, y0=3, y1=4, z0=-1, z1=-2)

        # 카메라 위치 텍스트 출력
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # 화면 표시 및 파일 저장
    cv.imshow('Pose Estimation (Staircase)', img)
    out_video.write(img)
    
    if cv.waitKey(1) == 27: # ESC 키를 누르면 중단
        break

# 자원 해제
video.release()
out_video.release()
cv.destroyAllWindows()

print(f"영상 처리가 완료되었습니다! '{output_video_file}'를 확인해보세요.")