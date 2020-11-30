################################## 1
# import matplotlib.pyplot
# matplotlib.pyplot.plot()
# matplotlib.pyplot.show()
#
# from matplotlib.pyplot import *
# plot()
# show()
#
import math

import matplotlib.pyplot as plt
# plt.plot()
# plt.show()
################################### 2
# types of plot
# 1. line plot
# 2. bar plot / bar chart
# 3. histogram plot
# 4. scatter plot
# 5. contour plot
# 6. surface plot
# 7. box plot
# 8. pie plot
#################################### 3
# plt.title("titleleleleleel")
# plt.show()
#################################### 4
# plt.plot([1, 4, 9, 16], [22, 49, 50, 70])
# plt.show()
#################################### 5
# plt.plot([1, 4, 9, 16], [22, 49, 50, 70])
# plt.grid(True)
# plt.show()
#################################### 6
# plt.plot([1, 4, 9, 16], [22, 49, 50, 70])
# plt.xlabel("Tiger")
# plt.ylabel("Lion")
# plt.show()
#################################### 7 hold
# plt.plot([22, 49, 50, 70], [1, 4, 9, 16])
# plt.plot([22, 49, 50, 70], [3, 6, 15, 36])
# plt.show()
#################################### 8
# plt.plot([10, 20, 30, 40], range(4))
# plt.plot([10, 20, 30, 40], range(2, 6))
# plt.plot(range(1, 5), "r")
# plt.plot(range(3, 7), "b")
# plt.plot(range(5, 9), "g")
# plt.plot(range(7, 11), "c")
# plt.plot(range(9, 13), "m")
# plt.plot(range(11, 15), "y")
# plt.plot(range(13, 17), "k")
# plt.plot(range(15, 19), "p")
# plt.plot(range(17, 21), "d")
# plt.plot(range(19, 23), "h")
# plt.plot(range(21, 25), "o")
# plt.plot(range(23, 27), "x")
# plt.plot(range(25, 29), "w")
#
# plt.show()
# a = '.,ov^<>w1234sp*hH+xDd'
# for k, v in enumerate( a ):
#     plt.plot( range( 1+k, 5+k ), v + '-' )
# plt.show()
#################################### 9
# plt.plot(range(1, 5), '-')      # 실선
# plt.plot(range(2, 6), '--')     # 반 점선
# plt.plot(range(3, 7), '-.')     # 파선
# plt.plot(range(4, 8), ':')      # 점선
# plt.plot(range(5, 9), 'gv:')    # 색상, 마커, 선 스타일 순으로 한 문자열로 설정, 공백 없어야함
# plt.show()
#################################### 10
# plt.plot(
#     [10, 20, 30, 40],  # x축
#     [1, 4, 9, 16], # y축
#     c="b",    # 선 색깔
#     lw=1,     # 선 굵기
#     ls="--",      # 선 스타일
#     marker="o",       # 마커 종류
#     ms=15,        # 마커 크기
#     mec="g",      # 마커 선 색깔
#     mew=2,        # 마커 선 굵기
#     mfc="r"       # 마커 내부 색깔
# )
# plt.show()
#################################### 11 (수업빠진날)

# x, y 축의 유효범위 설정, 그래프의 유효 범위에서 그림에서 나타내는 부분을 설정
# plt.plot([-20, 0, 50], [-30, 10, 70])
# plt.xlim(-10, 30)
# plt.ylim(-20, 50)
# plt.grid(True)
# plt.show()
# ticks x, y 축의 눈금자 생성 range(0, 60, 10) = [0, 10, ... 50], 축의 리스트 내 숫자들이 눈금자로 생성됨
# plt.xticks(range(0, 60, 10))
# plt.yticks(range(0, 60, 10))
# plt.plot([0, 50], [10, 70])
# plt.show()
# figure 창의 크기를 설정, 항상 맨위에 해야함
# plt.figure(figsize=(8, 6))
#
# # label 은 범례명
# plt.plot([1, 2, 3, 4], label='tiger')
# plt.plot([2, 3, 4, 1], label='lion')
#
# # [1, 2, 3, 4] 눈금을 다음 리스트 내 문자열로 변경, rotation = 각도 문자열을 회전하여 보여줌 (문자열이 길어 겹칠 때)
# plt.xticks([1, 2, 3, 4], ['monday', 'tuesday', 'wendsday', 'sunday'], rotation=70)
# plt.yticks([1, 2, 3, 4], ["hi", "hello", "bye", "eve"])
#
# # legend() : plot 의 범례명 표시하기, loc = 범례 위치 지정, 숫자별 위치
# # 2  9  1
# # 6     7
# # 3  8  4
# plt.legend(loc=1)
# # plt.gca().set_facecolor('g') : 그래프만 배경 색 지정, ' ' 내에 16진수, 0~255, 0~255에 비례한 값0.0~1.0, 또는 b, r, g 등의 문자
# # 칼라 값 간의 곱은 1 이상은 늘어나지만 비례 값은 소수점 곱으로 줄어들음
# plt.gca().set_facecolor('0.5')
#
# plt.show()
# 한글 설정
# import matplotlib.font_manager as fm
# font_location = 'C:/Windows/Fonts/malgun.ttf'   # 오른쪽 마우스 정보 얻음
# font_family = fm.FontProperties( fname=font_location).get_name()
#
# # plt.rc 폰트 설정, family 옵션으로 font_family 변수에 담긴 폰트로 설정
# plt.rc( 'font', family = font_family )
#
# plt.plot(['개', '소', '말', '양'], ['가', '나', '다', '라'])
# plt.show()
# 하나의 창에 여러 그래프를 출력
# subplot(a, b, c)  a는 행 개수, b는 열 개수, c는 위치 (0,0) 위치부터 1 시작
# plot 그래프들이 그 코드 위의 subplot 위치의 그래프에 나타남
# plt.subplot(4, 3, 1)
# plt.plot([1, 2, 3, 4], [30, 40, 50, 60])
# plt.subplot(4, 3, 2)
# plt.plot([1, 2, 3, 4], [30, 40, 50, 60])
# plt.show()
# 두 그래프를 양 쪽에 단위를 보여주는 그래프, x 축이 기준
# age = [10, 20, 30, 40]
# weight = [40, 50, 60, 70]
# height = [160, 170, 170, 160]
#
# # twinx() 를 두 plot 사이에 위치
# plt.plot(age, weight, 'b', label='weight')
# plt.twinx()
# plt.plot(age, height, 'g', label='height')
# # plt.gcf() 창을 matplotlib.figure.Figure 타입으로 a 에 저장
# a = plt.gcf()
# plt.show()
#
# # a를 그림파일로 저장
# a.savefig('test.png')
# plot 는 선 그래프, bar 는 막대 그래프
# plt.plot([1, 2, 3], [2, 3, 1])
# plt.bar([1, 2, 3], [2, 3, 1])
# plt.show()
# barh() 막대 그래프를 가로로 보여 줌
# plt.barh([1, 2, 3], [2, 3, 1])
# plt.show()
import mp as mp
from cv2.datasets import z
#
# a = ['python', 'c++', 'java', 'scala', 'lisp', 'perl', 'javascript']
# b = [10 , 12, 6 , 8, 11, 15, 4]
# align 옵션 center / edge, x 축의 눈금이 막대의 가운데에 오거나 막대의 가장 왼쪽에 오게 함
# plt.subplot(1, 2, 1)
# plt.bar(a, b, align='edge')
# plt.xticks(rotation=40)
# plt.subplot(1, 2, 2)
# plt.bar(a, b, align='center')
# plt.xticks(rotation=40)
# plt.show()
# 세번째 인수 : 막대 넓이, 네번째 인수 : bottom = (y 축 시작 수치) y 축 값이 bottom 수치에 더한 값만큼 나옴
# alpha 는 배경색과 막대색을 섞은 비율, 1이면 막대색과 같음
# plt.bar(a, b, 0.2, 20, align='edge', alpha=0.1)
# plt.show()
# numpy.arange
# a = [0, 1, 2, 3]
#a = a + 3     # list와 int는 + 연산이 불가능
# a = a + range(3)  # list와 range는 + 연산이 불가능
# np.arange(x)  x = 4 이면 0~3 까지의 원소 생성, 1, 4 이면 1부터 3까지 생성, 1, 10 , 2 이면 1부터 시작 간격2로 10 전까지 생성
# list와 np.arange() 를 더하면 numpy.ndarray 타입으로 바뀜(원소 사이 , 가 없음)
# 덧셈은 두 리스트의 원소 개수가 같아야함
# a = a + np.arange(4)
# print(a)
# np.arange(4)+0.0 은 x 축 값, 막대의 중앙이 해당 x축 값에 오게 됨, 넓이는 0.4로 설정
# a = plt.bar( np.arange(4)+0.0, [90, 55, 40, 65], 0.4 )
# b = plt.bar( np.arange(4)+0.2, [65, 40, 55, 95], 0.4 )
# plt.show()
# bottom 에 []로 x 축 리스트 길이와 같은 수를 넣으면 각 리스트 값만큼 시작하는 값이 달라짐
# a = plt.bar( np.arange(4), [90, 55, 40, 65], 0.4 )
# b = plt.bar( np.arange(4), [65, 40, 55, 95], 0.4, bottom=[90, 55, 40, 65] )     # 위 막대 그래프 위에 이어 붙여짐
# plt.show()
# 연도별 석탄 채굴량을 그래프로 그림
# import matplotlib.font_manager as fm
# font_location = 'C:/Windows/Fonts/malgun.ttf'   # 오른쪽 마우스 정보 얻음
# font_family = fm.FontProperties( fname=font_location).get_name()
# # plt.rc 폰트 설정, family 옵션으로 font_family 변수에 담긴 폰트로 설정
# plt.rc( 'font', family = font_family )
# plt.bar(range(1900, 2000, 10), [random.randint(100, 1000) for _ in range(10)], 10, color='lightblue', edgecolor='black')
# plt.xticks(range(1900, 2000, 10))
# plt.xlabel("연도")
# plt.ylabel("석탄 채굴량 (단위:t)")
# plt.show()
# 나이대 별 급여, 직업 별 급여, 성별 급여, 학과별 경쟁류르 월별 수익률, 팀별 성적, 종교 유무별 이혼률, 동물원별 사자와 호랑이에 대한 개체수,
# 나라별 남자 여자에 대한 인구수, 연령대별 성별에 대한 월급 차이, 성별 직업에 대한 빈도수(선호도), 지역별 연령대 비율, 연령대별 정당에 대한 지지율,
# 지역별 정당에 대한 지지율, 나라별 금은동에 대한 메달 수, 도시별 성별에 대한 범죄율
# 경제성장률, 자살률, 합병률, 취업률, 백분률(타율 등), 판매량, 재고량, 성장률, 생산량, 분포량, 빈도, 유입량, 이자율



#################################### 12 (수업빠진날)

import pandas as pd
# 1차원적인 자료는 series 로 표현
# 2차원적인 자료는 data frame 으로 표현
# data = {
#     '이름': ['김', '이', '박'],
#     '나이': [10, 20, 30],
#     '고향': ['서울', '부산', '대전']
# }
# DataFrame 생성
# df = pd.DataFrame(data)
# df['필드명'] 으로 특정 열 출력 가능
# print(df)
# a = [10, 20, 30, 40]    # 많은 부분에서 필드의 성격을 지닌다
# b = [50, 60, 70, 80]
# c = [a, b]
# x축이 index y축이 각 리스트(a, b)
# print(pd.DataFrame(c))
# 위 테이블의 x, y축을 반대로 나타냄
# df = pd.DataFrame(c).T
# df.columns 는 필드명 리스트, 다른 필드명 리스트를 대입하여 수정할 수 있음, col 개수와 맞아야함
# df.columns = ['나이', '이름']
# print(df)
# DataFrame 에 데이터는 입력하지 않고 필드명만 설정
# df = pd.DataFrame(columns=['이름', '나이', '고향'])
# len 으로 dataframe 을 입력하면 row 의 개수를 리턴
# print(len(df))
# df.loc[]  [20] 은 데이터의 키 값, row 의 키
# df.loc[20] = ['호랑이', 20, '서울']
# df.loc[30] = ['사자', 30, '대전']
# df.loc['??'] = ['hi', 40, 'bye']
# df.loc[30] = ['개', 10, '부산']    # 이미 키 값이 존재하는 경우 갱신하게 됨
# print(df)
# df = pd.DataFrame(columns=['이름', '나이', '고향'])
# for i in range(10):
#     df.loc[len(df)] = ['이순신' + str(i), 10 + i, '서울' + str(i)]
# row 삭제, df.drop 은 인수로 row 의 키 값을 받고 그 row 를 삭제한 DataFrame을 리턴함, 원본은 수정되지 않음
# 인수는 list 로 받음, list 안에 해당하는 모든 키의 row 를 삭제함
# df = df.drop([5, 7, 0])
# print(df)
# 원하는 row 출력
# print(df.loc[3])
# df의 인덱스상 3번째부터 4번째까지 출력함(index : 2, 3)
# print(df[2:4])
# =======================================================
# DataFrame.head(X)는 위에서 x개만 출력함, default 는 5
# print(df.head(3))
# DataFrame.tail(x)는 아래서 x개만 출력함, default 는 5
# print(df.tail(3)); print("=" * 30)   # separate 한 줄로 처리

#################################### 13
# 구구단
# def f1():
#     for i in range(10):
#         for j in range(10):
#             print(i , '*', j, '=', i*j, ' ', end='')
#         print('')
# # 합산
# def f2(a, b):
#     print('a+b =', a+b)
# a = {10: f1(), 20: f2(5, 3)}
# a[10]
# a[20]
#
# def f4():
#     print('합산')

# print(False ^ False)
# print(False ^ True)
# print(True ^ True)
# print(True ^ False)

import numpy as np
# a = np.arange(1000)
# print(a)
# b = list(range(100))
# print(b)
# c = a * 2
# print(c)
# d = [x * 2 for x in a]
# print(d)

# a = np.array([1, 2],
#              [3, 4],
#              [5, 6])
# b = np.array([2, 3],
#              [4, 5],
#              [6, 7])
# print(a + b); print(sep="*")

# a = np.array([1, 2, 3, 4, 5])
# b = np.array([2, 5, 4, 2, 10])
# print(a + b); print("=" * 20)
# print(a - b); print("=" * 20)
# print(a * b); print("=" * 20)
# print(a / b); print("=" * 20)
# print(a // b); print("=" * 20)
# print(a % b); print("=" * 20)
# print(1 / a); print("=" * 20)   # 역수로 생성
#
# 4**0.5
# a = np.arange(10)
# print(a[:7])
# print(a[2:7])
# print(a[6:])
# print(a[:])

# a = np.empty((3, 4))
# b = np.empty((4, 2))
# print(np.dot(a, b))

# a = np.arange(10)
# print(a)
# b = np.arange(10, 8, -1)
# print(b)
# print(np.sqrt(a))

# a = np.sign(1, 20, -2)

# a = np.array((1, 2, 3))
# b = np.array((2, 3, 4))
# print(np.add(a, b))

# points = np.arange(-5, 5, 0.01)
# xs, ys = np.meshgrid(points, points)
#
# z = np.sqrt(xs ** 2 + ys ** 2)
#
# import matplotlib.pyplot as plt
# plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# plt.show()

# a = np.array([1.1, 1.2, 1.3, 1.4 , 1.5])
# b = np.array([2.1, 2.2, 2.3, 2.4 , 2.5])
# c = np.array([True, False, True, True, False])
# result = [(x if c else y)
#           for x, y, c in zip(a, b, c)]
# print(result)
# result = np.where(c, a, b)
# print(result)

# a = np.random.randn(5, 4)
# print(a)
# print(a.mean())
# print(np.mean(a))
# print(a.sum()/len(a))
# print(np.count_nonzero(a[:]))

# arr = np.arange(10)
# np.save('some_array', arr)
# print(np.load('some_array.npy'))
# np.savez('array_archive.npz', a=arr, b=arr)
# arch = np.load('array_archive.npz')
# print(arch['a'], arch['b'])

#################################### 14
import numpy as np
# a = np.array([1, 2, 3])
# # 대각이 1, 2, 3 이고 나머지가 0 인 단위행렬 리턴
# b = np.diag(a)
# print(b)
# # 대각의 합
# print(np.trace(b))

# unit_mat_4 = np.eye(4)
# print(unit_mat_4)

# x = np.arange(9).reshape(3, 3)
# print(x)
# print(np.diag(x))
# print(np.diag(np.diag(x)))

################################ 여기서 왜 내적 곱 값이 저렇게 나오는가
# a = np.arange(4).reshape(2, 2)
# print(a)
# print(a*a)
# print(np.dot(a, a))
# print(a.dot(a))

#################################trace는 대각원소의 합
# b = np.arange(16).reshape(4, 4)
# print(b)
# print(np.trace(b))

# c = np.arange(27).reshape(3, 3, 3)
# print(c)
# print(np.trace(c))

# d = np.array([[1, 2], [3, 4]])
# print(np.linalg.det(a))

# a = np.array(range(4)).reshape(2, 2)
# print(a)
# a_inv = np.linalg.inv(a)
# print(a_inv)

# a = np.array([
#     [2, 3],
#     [3, 1]
# ])
# b = np.array([
#     [5],
#     [2]
# ])
# # a * x = b
# x = np.linalg.solve(a, b)
# print(x)

# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])
# A = np.vstack([x, np.ones(len(x))]).T
# print(A)
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]
# print(m, c)
#
# import matplotlib.pyplot as plt
# plt.plot(x, y, 'o', label='Original data', markersize=10)
# plt.plot(x, m*x + c, 'r', label='Fitted line')
# plt.legend()
# plt.show()
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# fig, ax = plt.subplots()
#
# ax.plot([1,4],[1,4])
#
# ax.add_patch(
#      patches.Rectangle(
#         (1, 1),
#         0.5,
#         0.5,
#         edgecolor = 'blue',
#         facecolor = 'red',
#         fill=True
#      ) )
# 사각의 좌표
# 4 3
# 1 2
# xy = [1, 2, 3, 4, 1]
# 사각의 좌표 xy = [(x1, y1), (x2, y2) ... ]

# def move(xy):
#     dx = 50; dy = 30
#     # a는 이동 행렬
#     a = np.array([
#         [1, 0, dx],
#         [0, 1, dy],
#         [0, 0, 1]
#     ])
#     # b는 xy좌표를 행렬로
#     x_values = []
#     y_values = []
#     for i in xy:
#         b = np.array([
#             [i[0]],
#             [i[1]],
#             [1]
#         ])
#         mul_result = a @ b
#         x_values.append(mul_result[0][0])
#         y_values.append(mul_result[1][0])
#
#     return x_values, y_values
#
#
# xy = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
#
# x_list = [i[0] for i in xy]
# y_list = [i[1] for i in xy]
#
# move_result = move(xy)
#
# print(move_result, type(move_result))
# plt.xlim(-20, 100)
# plt.ylim(-20, 100)
# plt.plot(x_list, y_list)
# plt.plot(move_result[0], move_result[1])
# plt.show()

# ========================================
# 개체 회전
# def rot(xy):
#     Theta = 30
#     # a는 회전 행렬
#     cosTH = math.cos(math.radians(Theta))
#     sinTH = math.sin(math.radians(Theta))
#     a = np.array([
#         [cosTH, -sinTH, 0],
#         [sinTH, cosTH, 0],
#         [0, 0, 1]
#     ])
#     print(a)
#     # b는 xy좌표를 행렬로
#     x_values = []
#     y_values = []
#     for i in xy:
#         b = np.array([
#             [i[0]],
#             [i[1]],
#             [1]
#         ])
#         mul_result = a @ b
#         print(mul_result)
#         x_values.append(mul_result[0][0])
#         y_values.append(mul_result[1][0])
#
#     return x_values, y_values
#
#
# xy = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
#
# x_list = [i[0] for i in xy]
# y_list = [i[1] for i in xy]
#
# rot_result = rot(xy)
#
# print(rot_result, type(rot_result))
# plt.xlim(-20, 100)
# plt.ylim(-20, 100)
# plt.plot(x_list, y_list)
# plt.plot(rot_result[0], rot_result[1])
# plt.show()

#################################### 15
# import random
# position = 0
# walk = [position]
# steps = 1000
# for i in range(steps):
#     step = 1 if random.randint(0, 1) else -1
#     position += step
#     walk.append(position)
# plt.plot(walk[:100])
# plt.show()
#
# nsteps = 1000
# draws = np.random.randint(0, 2, size=nsteps)
# steps = np.where(draws > 0, 1, -1)
# walk = steps.cumsum()
#
# walk.min()
# walk.max()
# (np.abs(walk) >= 10).argmax()

#################################### 16
# obj = pd.Series([4, 7, -5, 3])
# print(obj)
# print(obj.values)
# print(obj.index)
# obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# print(obj2)
# print(obj2.index)

#################################### 17
# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2000, 2001, 2002, 2001, 2002, 2003],
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data)

#################################### 18

# df = pd.DataFrame({'t1': [2, 3, 6, 8], 't2': [81, 93, 91, 97]})
# print(df)
# print(df.corr(method='pearson'))
# print(df['t1'].corr(df['t2']))

# df = pd.DataFrame( columns=[ 't1', 't2' ] )
# print(df.columns)
# df.loc['0'] = [2, 81]
# df.loc['1'] = [4, 93]
# df.loc['2'] = [6, 91]
# df.loc['3'] = [8, 97]
# print(df)
# print(df.corr(method='pearson')) # 문제의 코드
# print(df['t1'].corr(df['t2'])) # 요것이 에러

# print(pd.read_csv('mytxt/ex1.csv'))
#
# print(pd.read_csv('mytxt/ex1.csv', header=None))
#
# names = ['a', 'b', 'c', 'd', 'message']
# print(pd.read_csv('mytxt/ex1.csv', names=names, index_col='message'))


#################################### 19
import pandas as pd
from pandas import Series, DataFrame
# obj = pd.Series([4, 7, -5, 3])
# print(obj)
# print(obj.values)
# print(obj.index)
# obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# print(obj2)
# print(obj2.index)
# print(obj2['a'])
# obj2['d'] = 6
# print(obj2[['c', 'a', 'd']])
# print(obj2[obj2 > 0])
# print(obj2 * 2)
# print(np.exp(obj2))
# print('b' in obj2)

# sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
# obj3 = pd.Series(sdata)
# print(obj3)
# states = ['California', 'Ohio', 'Oregon', 'Texas']
# obj4 = pd.Series(sdata, index=states)
# print(obj4)

# print(pd.isnull(obj4))
# print(pd.notnull(obj4))
# print(obj4.isnull())
# print(obj3)
# print(obj4)
# print(obj3 + obj4)
# obj4.name = 'population'
# obj4.index.name = 'state'
# print(obj4)
# print(obj)
# obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
# print(obj)
# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
#         'year': [2000, 2001, 2002, 2001, 2002, 2003],
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data)
# print(frame)
# print(frame.head())
# print(pd.DataFrame(data, columns=['year', 'state', 'pop']))
# frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
#                       index=['one', 'two', 'three', 'four', 'five', 'six'])
# print(frame2)
# print(frame2.columns)
# print(frame2['state'])
# print(frame2.year)
# print(frame2.loc['three'])
# frame2['debt'] = 16.5
# print(frame2)
# frame2['debt'] = np.arange(6.)
# print(frame2)
# val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
# frame2['debt'] = val
# print(frame2)
# frame2['eastern'] = frame2.state =='Ohio'
# print(frame2)
# del frame2['eastern']
# print(frame2.columns)
#
# pop = {'Nevada': {2001: 2.4, 2002: 2.9},
#        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# frame3 = pd.DataFrame(pop)
# print(frame3)

# data = {'Tiger': [10, 20, 30], 'Lion': [20, 40, 60]}
# frame = DataFrame(data)
# print(frame)
#
# fn = 'out.csv'
# frame.to_csv(fn)
#
# fn = '호%d랑%d이' % (10, 20)
# print(fn)
# fn2 = '%d.csv' % 1
# print(fn2)
#
# for i in range(10):
#     fn = 'mytxt/Tiger%04d.csv' % i
#     data.to_csv(fn)
#     print(fn)

# import sys
# a = data.to_csv(sys.stdout, sep='|')
# print(a)
#
# data = {'Tiger': [10, 20, None], 'Lion': [20, 40, 60]}
# frame = DataFrame(data)
# b = data.to_csv(sys.stdout, na_rep='Tiger')
# print(b)

# import sys
# a = data.to_csv(sys.stdout, sep='|')
# print(a)
# data = pd.DataFrame({'호랑이': [10, None, 30, 40],
#                        '사자': [15, 25, 35, 45]})
# b = data.to_csv(sys.stdout, na_rep='TIGER')
# print(b)
# c = pd.read_csv('mytxt/0000.csv')   # null값이 있는 파일을 불러오면 어떻게 될까
# print(c)
# a = [1, 2, 3]
# b = [[4, 5, 6]]
# for i in zip(a, *b):
#     print(i)
#################################### 20
obj = """
{"name": "Wes",
   "places_lived": ["United States", "Spain", "Germany"],
   "pet": null,
   "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
                {"name": "Katie", "age": 38,
                 "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
# import json
# result = json.loads(obj)
# print(result)
# asjson = json.dumps(result)
# siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
# print(siblings)



#################################### 21

# import requests
# url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
# resp = requests.get(url)
# print(resp)
# data = resp.json()
# print(data[0]['title'])

#################################### 22

# string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
# print(string_data)
# print(string_data.isnull())
# string_data[0] = None
# print(string_data.isnull())
# from numpy import nan as NA
# data = pd.Series([1, NA, 3.5, NA, 7])
# print(data.dropna())
# print(data[data.notnull()])
# data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
#                     [NA, NA, NA], [NA, 6.5, 3.]])
# cleaned = data.dropna()
# print(data)
# print(cleaned)
# data[4] = NA
# print(data)
# print(data.dropna(axis=1, how='all'))
# df = pd.DataFrame(np.random.randn(7, 3))
# df.iloc[:4, 1] = NA
# df.iloc[:2, 2] = NA
# print(df)
# print(df.dropna())
# print(df.dropna(thresh=2))
#
# print(df.fillna(0))
# print(df.fillna({1: 0.5, 2: 0}))
# _ = df.fillna(0, inplace=True)
# print(df)

#################################### 23

data = pd.DataFrame(np.arange(6).reshape(2, 3),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'], name='number'))
print(data)
# print(pd.DataFrame(np.arange(6).reshape(2, 3),
#                     index=pd.Index(['Ohio', 'Colorado'], name='state'),
#                     columns=pd.Index(['one', 'two', 'three'], name='number')))

result = data.stack()
print(result)

print(result.unstack())


#################################### 24
#################################### 25
#################################### 26
#################################### 27
#################################### 28