import pandas as pd
import operator
import datetime
import numpy as np

# 날짜 빠진거 있는지 관측 지점별로 groupby
df = pd.read_csv('20190101_20191231_농진청 기상 일별 자료(2019-10-23).xlsx - 기상관측 파일데이터 일별 자료.csv', skiprows=1, nrows=0)
# print(df)
ef = pd.read_csv('20190101_20191231_농진청 기상 일별 자료(2019-10-23).xlsx - 기상관측 파일데이터 일별 자료.csv', skiprows=1)
print(ef)
ef_group = ef.groupby('관측지점')

count = 0
for i in ef_group:
    group_df = i[1]
    df_dates = list(group_df['조회일자'])

    for j in range(len(df_dates)):

        date_list = df_dates[j].split('-')
        year = int(date_list[0])
        month = int(date_list[1])
        day = int(date_list[2])
        date_time = datetime.datetime(year, month, day)

        try:
            next_date_list = df_dates[j + 1].split('-')
        except:
            next_date_list = df_dates[j].split('-')

        n_year = int(next_date_list[0])
        n_month = int(next_date_list[1])
        n_day = int(next_date_list[2])
        next_date_time = datetime.datetime(n_year, n_month, n_day)

        if next_date_time - date_time > datetime.timedelta(days=1):
            print('빠진 일자 존재\n', i[1]['관측지점'],"\n", date_time, next_date_time)
            count = count + 1


print(count)
    # next_date_time = datetime.datetime(year, month, day) + datetime.timedelta(days=1)
