######################
# 2.15 True and False
######################
"""
a = 10
print(a if a is not None else 0)

a = None
print(a if a is not None else 0)
"""
######################
# 2.16 정렬
######################
import math
from typing import List

from tensorflow.core.kernels.boosted_trees.boosted_trees_pb2 import Vector

"""
x = [4, 1, 2, 3]
y = sorted(x)
x.sort()

x = sorted([-4, 1, -2, 3], key=abs, reverse=True)

wc = sorted(word_counts.items(),
            key=lambda word_and_count: word_and_count[1], reverse=True)
"""

######################
# 2.17 리스트 컴프리헨션
######################
"""
even_numbers = [x for x in range(5) if x % 2 == 0]
squares = [x * x for x in range(5)]
even_squares = [x * x for x in even_numbers]
square_dict = {x: x * x for x in range(5)}
square_set = {x * x for x in [1, -1]}
zeros = [0 for _ in even_numbers]
pairs = [(x, y)
         for x in range(10)
         for y in range(10)]
increasing_pairs = [(x, y)
                    for x in range(10)
                    for y in range(x + 1, 10)]
"""
"""
a = [10 for _ in [1, 1, 1, 1]]
print(a)

a = [(i, j)
     for i in range(3)
     for j in range(4)
     ]
print(a)

for i in range(3):
    for j in range(4):
        print(a[i*4+j], end=' ')
    print()
"""
##########################
# 2.18 자동 테스트와 assert
##########################
"""
assert 1 + 1 == 2
def smallest_item(xs):
    return min(xs)
assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1
"""
##############################
# 2.19 객체 지향 프로그래밍(OOP)
##############################
"""
class CountingClicker:
    def __init__(self, count=0):
        self.count = count
clicker1 = CountingClicker()
clicker2 = CountingClicker(100)
clicker3 = CountingClicker(count=100)

def __repr__(self):
    return f"countingClicker(count={self.count})"
def click(self, num_times = 1):
    # 한 번 실행할 때마다 num_times 만큼 count 증가
    self.count += num_times
def read(self):
    return self.count
def reset(self):
    self.count = 0

clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"

class NoResetClicker(CountingClicker):
    def reset(self):
        pass
clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.click()
assert clicker2.read() == 1, "reset shouldn't do anything"
"""
"""
class Apple:
    def __init__(self, count=0):
        self.count = count
a1 = Apple()
a2 = Apple(10)
a3 = Apple(count=20)

class Apple:
    def __init__(self, count=0):
        self.count = count
    def f1(self):
        print(self.count)
a1 = Apple()
a1.f1()
a2 = Apple(10)
a2.f1()
a3 = Apple(count=20)
a3.f1()
"""


############################
# 2.20 이터레이터와 제너레이터
############################
"""
def generate_range(n):
    i = 0
    while i < n:
        yield i
        i += 1
for i in generate_range(10):
    print(f"i: {i}")
def natural_numbers():
    # 1, 2, 3, ... 을 반환
    n = 1
    while True:
        yield n
        n += 1
evens_blow_20 = (i for i in generate_range(20) if i % 2 == 0)
data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)

names = ["Alice", "Bob", "Charlie", "Debbie"]
for i, name in enumerate(names):
    print(f"name {i} is {name}")
"""
"""
def func1():
    yield 10
    yield 20
    yield 30
for i in func1():
    print(i)
def func2():
    for i in func2(10):
        print(i)
a = func1()
print(a)
for i in func1():
    print(i)
b = (i for i in a i % 2 == 0)
print(b)
"""
################
# 2.21 난수 생성
################
"""
import random

random.seed(10)
four_uniform_randoms = [random.random() for _ in range(4)]
random.seed(10)
print(random.random())
random.seed(10)
print(random.random())
random.randrange(10)
random.randrange(3, 6)
up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)
my_best_friend = random.choice(["Alice", "Bob", "Charlie"])
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)
"""
"""
print(random.random())
print(random.randint(0, 10))
print(random.randrange(0, 10, 3))
random.seed(10)

for i in range(5):
    print(random.random())
    random.seed(10)

a = [1, 2, 3, 4, 5]
random.shuffle(a)
print(a)
b = random.choice(a)
print(b)
c = random.sample(a, 2)
print(c)
"""
################
# 2.22 정규표현식
################
"""
import re
re_example = [
    not re.match("a", "cat"),
    re.search("a", "cat"),
    not re.search("c", "dog"),
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]", "-", "R2D2")
]
assert all(re_example), "all the regex examples should be True"
"""
################
# 2.23 함수형 도구
################
#######################
# 2.24 zip 과 인자 언패킹
#######################
"""
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
[pair for pair in zip(list1, list2)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
"""
####################
# 2.25 arg와 kwargs
####################
#####################
# 2.26 타입 어노테이션
#####################
"""
def add(a, b):
    return a + b
def add(a: int, b: int) -> int:
    return a + b
add(10, 5)
add("hi", "there")
def dot_product(x, y): ...
"""
##############################
# 2.26.1 타입 어노테이션하는 방법
##############################
"""
a: list = []
from typing import Callable, List

b: List = []
c: List[int] = []
# d: list[int] = [] # 3.8version 터짐, 3.9 안터짐
"""
"""
def total(xs: list) -> float:
    return sum(total)
def total(xs: List[float]) -> float:
    return sum(total)
values = []
best_so_far = None

from typing import Optional
values = []
best_so_far: Optional[float] = None
"""
################################################
            # Chapter 3 데이터 시각화
################################################

#################
# 3.1 matplotlib
#################

from matplotlib import pyplot as plt
"""
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5,  5979.6, 10289.7, 14958.3]

plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
plt.title("Nominal GDP")
plt.ylabel("Billions of $")
plt.show()
"""
#################
# 3.2 bar charts
#################
"""
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

plt.bar(range(len(movies)), num_oscars)
plt.title("My Favorite Movies")
plt.ylabel("# of Academy Awards")
plt.xticks(range(len(movies)), movies)
plt.show()
"""
from collections import Counter
"""
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
print(grades[1] / 10)
print(grades[1] // 10)
print((grades[1] // 10) * 10)
print(Counter(min(grade // 10 * 10, 90) for grade in grades))
for grade in grades:
    print((min((grade // 10) * 10, 90)), end=' ')
print()
print('histogram :', end=" ")
print(histogram)

plt.bar([x + 5 for x in histogram.keys()],
        histogram.values(),
        10,
        edgecolor=(0, 0, 0))

print([x + 5 for x in histogram.keys()])
print(histogram.values())

plt.axis([-5, 105, 0, 5])
plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
"""

"""
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")
plt.ticklabel_format(useOffset=False)
# plt.axis([2016.5, 2018.5, 499, 506]) # 비정상적인 y축 그래프
plt.axis([2016.5, 2018.5, 0, 550]) # 정상적인 y축 그래프
plt.title("Look at the 'Slight' Increase!")
plt.show()
"""

##################
# 3.2 line charts
##################
"""
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]

total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

plt.plot(xs, variance, 'g-', label='variance')
plt.plot(xs, bias_squared, 'r-.', label='bias^2')
plt.plot(xs, total_error, 'b:', label='total error')

plt.legend(loc=9)   # automatic because assigned labels
plt.xlabel("model complexity")
plt.title("The Bias-Variance Tradeoff")
plt.show()
"""
###################
# 3.2 scatterplots
###################
"""
friends = [78, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

for label, friend_count, minute_count in zip(labels, friends, minutes):
	plt.annotate(label,
		xy=(friend_count, minute_count),
		xytext=(5, -5),
		textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
#if you're scattering comparable variables do plt.axis("equal")
plt.show()
"""
################################################
            # Chapter 4 선형대수
################################################
# 원점기준 <위치값> + 방향
###################
# 4.1 Vector
###################
"""
def f1(a, b):
    c = []
    c.append(a[0] + b[0])
    c.append(a[1] + b[1])
    c.append(a[2] + b[2])
    return c
print(f1([1, 2, 3], [4, 5, 6]))

def f1(a, b):
    result = [i + j for i, j in zip(a, b)]
    return result
print(f1([1, 2, 3], [4, 5, 6]))

def f1(v: Vector, w: Vector) -> Vector:
    return [v_i - w_i for v_i, w_i in zip(v, w)]
print(f1([1, 2, 3], [4, 5, 6]))
#
def f0(vectors: List[Vector]) -> Vector:
    num_elements = len(vectors[0])
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]
# print(f1 ( [ [1, 2, 3], [4, 5, 6], [4, 5, 6] ] ) )
def f2(c: float, v: Vector):
    return[c * i for i in v]
print(f2(10, [1, 2, 3]))

def f1(vectors: List[Vector]):
    n = len(vectors)
    return f2(1/n, f0(vectors))
print(f1([[1,2,3], [2,3,4]]))

def f3(v: Vector, w: Vector):
    return sum(a * b for a, b in zip(v, w))
print(f3([1, 2, 3], [4, 5, 6]))


def f4(v: Vector):
    return f3(v, v)

import math

def f5(v: Vector):
    return math.sqrt(f4(v))
print(f5([3, 4]))

def f6(v: Vector):
    return (i, v)
"""
###################
# 4.2 행렬
###################
"""
Matrix = List[List[float]]
A = [[1, 2, 3],
     [4, 5, 6]]
B = [[1, 2],
     [3, 4],
     [5, 6]]

from typing import Tuple
def f0(A: Matrix):
    a = len(A)
    b = len(A[0]) if A else 0
    return a, b

print(f0([[1, 2, 3], [4, 5, 6]]))

def f1(A: Matrix, i: int):
    return A[i]
print(f1([[1, 2, 3],
          [4, 5, 6]],
         1))

def f2(A: Matrix, j: int):
    return [A_i[j]
            for A_i in A]
print(f2([[1, 2, 3],
          [4, 5, 6]],
         2))

from typing import Callable

                                #이거 확인필요
def f0(num_rows: int,
       num_cols: int,
       entry_fn: Callable[[int, int], float]):
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]
                                print(f0(5, 5, [[5, 5], 5]))
def f1(n: int):
    return f0(n, n, lambda i, j: 1 if i == j else 0)
                                print(f1(10))
a = [[1, 2, 3, 4],
     [2, 3, 1, 2],
     [1, 2, 1, 3]]

b = [[2, 1, 3],
     [1, 2 ,2],
     [2, 3, 1],
     [3, 4, 1]]

result = []

# 행렬의 열을 구하는 함수
def get_column(a, b):
    return [a_i[b] for a_i in a]
def mul_matrix(a, b):
    print(len(a[0]), len(b))
    assert len(a[0]) == len(b), "행렬 a, b에 대한 곱을 위해서는 a의 열의 개수와 b의 행의 개수가 같아야 합니다."
    for a_row in a:
        result_row = []
        for j in range(len(b[0])):
            b_col = get_column(b, j)
            result_row.append(sum(a_row_v * b_col_v
                                  for a_row_v, b_col_v
                                  in zip(a_row, b_col)))
        result.append(result_row)
mul_matrix(a, b)
print('=======')
for rows in result:
    print(rows)
"""
################################################
            # Chapter 5 통계
################################################
###################
# 5.1 데이터셋 설명하기
###################
"""
from collections import Counter
import matplotlib.pyplot as plt

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

a = Counter(num_friends)
xs = range(101)
ys = [a[x] for x in xs]
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
# plt.show()

num_points = (len(num_friends))
largest_value = max(num_friends)
smallest_value = min(num_friends)
sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
"""
###################
# 5.1.1 중심 경향성
###################
"""
def mean(xs: List[float]):
    return sum(xs) / len(xs)
# print(mean(num_friends))

def _median_odd(xs: List[float]):
    return sorted(xs[ len(xs) // 2 ])

def _median_even(xs:List[float]):
    hi_point = len(xs)//2
    low_point = hi_point - 1
    return (sorted(xs)[hi_point] + sorted(xs)[low_point]) / 2

def median(xs: List[float]):
    return _median_even(xs) if len(xs) % 2 == 0 else _median_odd(xs)

print(median(num_friends))

def f1():
    return 10
def f2():
    return f1()
def f3():
    return f2()
print(f3())

a = abs(-3)
print(a)
abs = 10
a = abs(-3)
print(a)
print(int(3.14))
print(int(0.2 * 260))

def quantile(xs:List[float], p: float) -> float:
    p_index: int = int(p * len(xs))
    return sorted(xs)[p_index]
# print(quantile(num_friends, 0.99))

def mode(xs: List[float]) -> float:
    print('xs : ', xs)
    counts = Counter(xs)
    print(counts)
    # max_count = counts.most_common(3)
    max_count = max(counts.values())
    print(sorted(counts.values()))
    print(max_count)
    return [i for i, count in counts.items() if count == max_count]
print(mode(num_friends))
"""
###################
# 5.1.2 산포도
###################
"""
a = [1, 2, 3, 4, 5]
print(a)    #변이
b = sum(a) / len(a)
print(b)    #평균
c = [i - b for i in a]
print(c)    #편차
d = sum(abs(i - b) for i in a) / len(a)
print(d)    #편차절대값의 합의 평균
f = math.sqrt(sum(abs(i - b) for i in a) / len(a))
print(f)    #표준편차

print(0 <= 3 <= 7)
"""

###################
# 5.2 상관관계
###################


###################
# 5.3 심슨의 역설
###################
####################################
# 5.4 상관관계에 대한 추가적인 경고 사항
####################################
#######################
# 5.5 상관관계와 인과관계
#######################
################################################
            # Chapter 6 확률
################################################
#######################
# 6.1 종속성과 독립성
#######################
#######################
# 6.2 조건부 확률
#######################
"""
from enum import Enum
import enum
import random
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1
print(Kid.BOY)
print(Kid.BOY.value)
print(Kid.BOY.name)
print(type(Kid.BOY))

random.choice([Kid.BOY, Kid.GIRL])

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

random.choice([Color.RED, Color.GREEN, Color.BLUE])

CITY0 = 0
CITY1 = 1
CITY2 = 2

for _ in range(1000):
    a = random.choice([Kid.BOY, Kid.GIRL])
    b = random.choice([Kid.BOY, Kid.GIRL])

    if a == Kid.BOY:
        CITY0 += 1
    if a == Kid.BOY and b == Kid.GIRL:
        CITY1 += 1
    if a == Kid.BOY or b == Kid.GIRL:
        CITY2 += 1

print('ct0 :', CITY0, 'ct1 :', CITY1, 'ct2 :', CITY2)
print(CITY1/CITY0)
print(CITY1/CITY2)

count = 0
for i in range(10000):
    a = random.choice(range(10))
    b = random.choice(range(10))
    while(True):
        if a == b:
            b = random.choice(range(10))
        if a != b:
            break
    # print(i, a, b)

for i in range(10000):
    c = random.choice(range(10))
    d = random.choice(range(10))
    while(True):
        if c == d:
            d = random.choice(range(10))
        if c != d:
            break
    # print(i, c, d)

if a == c or b == d or a == d:

    count += 1
    print("당첨")

else:
    print("꽝")
if a == d or b == c or b == d:

    count += 1
    print("당첨")

else:
    print("꽝")


print("Total count is : ", end=" ")
print(count)
"""
#######################
# 6.3 베이즈 정리
#######################
#######################
# 6.4 확률변수
#######################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.6 정규 분포
#######################
"""
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()
"""

################################################
            # Chapter 7
################################################

import numpy as np
# x = [2, 4, 6, 8]
# y = [81, 93, 91, 97]
#
# plt.scatter(x, y)
# # plt.bar(x, y)
# for i, j in zip(x, y):
# 	plt.annotate(i,
# 		xy=(i, j),
# 		xytext=(5, -5),
# 		textcoords='offset points')


# x - x 평균 * y - y 평균 의 합 // 나누기 x - x 의 평균 의 제곱의 합
# a = 모든 x, y 에 대하여 ((x- x의 평균) * (y - y의 평균))의 합     /    ((x-x의 평균)^2) 의 합

# a = sum(x)/len(x)   # 평균
# b = sum(y)/len(y)   # 평균
# # 회귀선 y = ax + b
# print(sum((i - a) * (j - b) for i, j in zip(x, y)) / sum((i - a)**2 for i in x))
# result_a = sum((i - a) * (j - b) for i, j in zip(x, y)) / sum((i - a)**2 for i in x)
# result_b = b - (a * result_a)
#
# plt.plot(x, [result_a * i + result_b for i in x])
#
# plt.title("The GRAPH")
# plt.xlabel("XX label")
# plt.ylabel("YY label")
# # plt.axis("equal")
# plt.show()
#
# x = [2, 4, 6, 8]
# y = [81, 93, 91, 97]
# z = [i * result_a + result_b for i in x]  # [83.6, 88.2, 92.8, 97.4] 예측값
# # (83.6 - 81) ** 2 + (88.2 - 93) ^ 2 + ... / n
# # (예측값 - 실제값)^2
# # c 는 예측 값
# c = [i * result_a + result_b for i in a]
# print(sum((i - j) ** 2 for i, j in zip(c, b)) / len(c))
# # 평균 제곱 법 : (예측값- 실제값)^2 의 평균
# print(sum((i - j) ** 2 for i, j in zip(c, b)) / len(c))
#
# print(z)
#
# y = ax + b
# 최소제곱: a, b
# 평균제곱: (예측값 - 실제값) / n = 평균
# 2차 방정식: y = ax ** 2 + bx + c
# 순간변화율 = 기울기
# 미분 = 순간 기울기
# print(x.mean())


################################################
            # Chapter 8 경사 하강법
################################################
#######################
# 8.1 경사 하강법에 숨은 의미
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

################################################
            # Chapter 7
################################################
#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

################################################
            # Chapter 7
################################################
#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

#######################
# 6.5 연속 분포
#######################

################################################
            # Chapter 7
################################################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################

################################################
            # Chapter 7
################################################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################
#######################
# 6.5 연속 분포
#######################
