from pyecharts.charts import Map, Geo
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import pandas as pd
from tkinter import *
import webbrowser

dataset = pd.read_csv('owid-covid-data.csv')

#change date from object data type to datetime data type
dataset['date'] = pd.to_datetime(dataset['date'])
df = dataset.sort_values(by=['date'], ascending=False) #sort data by date
map_df = df[df['date']=='2020-10-14']
map_df.reset_index(drop=True, inplace=True)
map_df

country = list(map_df['location'])
totalcases = list(map_df['total_cases'])

list1 = [[country[i], totalcases[i]] for i in range(len(country))]
map_1 = Map(init_opts=opts.InitOpts(width="1000px", height="460px"))
map_1.add("Total Confirmed Cases", list1, maptype='world')
map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False)) #remove country names
map_1.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=1100000,is_piecewise=False),
                      legend_opts=opts.LegendOpts(is_show=False))

webbrowser.open(map_1.render('image.html'))



# list1 = [[country[i],totalcases[i]] for i in range(len(country))]
# map_1 = Map(init_opts=opts.InitOpts(width="1000px", height="460px"))
# map_1.add("Total Confirmed Cases",
#  list1,
#  maptype='world',
#  is_map_symbol_show=False)
# map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
# map_1.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=1100000, is_piecewise=False),
#  legend_opts=opts.LegendOpts(is_show=False))
# map_1.render_notebook()

# list1 = [[country[i],totalcases[i]] for i in range(len(country))]
# map_1 = Map(init_opts=opts.InitOpts(width=1000, height=460))
# map_1.add("Total Confirmed Cases",
#  list1,
#  maptype='world',
#  is_map_symbol_show=False)
# map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
# map_1.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=1100000,is_piecewise=True,pieces=[
#  {"min": 500000},
#  {"min": 200000, "max": 499999},
#  {"min": 100000, "max": 199999},
#  {"min": 50000, "max": 99999},
#  {"min": 10000, "max": 49999},
#  {"max": 9999},]),
#  legend_opts=opts.LegendOpts(is_show=False))
# map_1.render_notebook()
#
# list1 = [[country[i],totalcases[i]] for i in range(len(country))]
# map_1 = Map(init_opts=opts.InitOpts(width="1000px", height="460px"))
# map_1.add('Total Confirmed Cases',
#  list1,
#  maptype='world',
#  is_map_symbol_show=False)
# map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
# map_1.set_global_opts(
# visualmap_opts=opts.VisualMapOpts(max_=1100000,is_piecewise=True,pieces=[
#  {"min": 500000},
#  {"min": 200000, "max": 499999},
#  {"min": 100000, "max": 199999},
#  {"min": 50000, "max": 99999},
#  {"min": 10000, "max": 49999},
#  {"max": 9999},]),
#  title_opts=opts.TitleOpts(
#  title='Covid-19 Worldwide Total Cases',
#  subtitle='Till July 05th,2020',
#  pos_left='center',
#  padding=0,
#  item_gap=2,# gap between title and subtitle
#  title_textstyle_opts= opts.TextStyleOpts(color='darkblue',
#  font_weight='bold',
#  font_family='Courier New',
#  font_size=30),
#  subtitle_textstyle_opts= opts.TextStyleOpts(color='grey',
#  font_weight='bold',
#  font_family='Courier New',
#  font_size=13)),
#  legend_opts=opts.LegendOpts(is_show=False))
# map_1.render_notebook()
#
list1 = [[country[i],totalcases[i]] for i in range(len(country))]
map_1 = Map(init_opts=opts.InitOpts(width="1000px", height="460px", theme=ThemeType.ROMANTIC))
map_1.add('Total Confirmed Cases',
 list1,
 maptype='world',
 is_map_symbol_show=False)
map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
map_1.set_global_opts(
visualmap_opts=opts.VisualMapOpts(max_=1100000,is_piecewise=True,pieces=[
 {"min": 500000},
 {"min": 200000, "max": 499999},
 {"min": 100000, "max": 199999},
 {"min": 50000, "max": 99999},
 {"min": 10000, "max": 49999},
 {"max": 9999},]),
 title_opts=opts.TitleOpts(
 title="Covid-19 Worldwide Total Cases",
 subtitle='Till July 03rd,2020',
 pos_left='center',
 padding=0,
 item_gap=2,
 title_textstyle_opts= opts.TextStyleOpts(color='darkblue',
 font_weight='bold',
 font_family='Courier New',
 font_size=30),
 subtitle_textstyle_opts= opts.TextStyleOpts(color='grey',
 font_weight='bold',
 font_family='Courier New',
 font_size=13)),
 legend_opts=opts.LegendOpts(is_show=False))
map_1.render_notebook()
