#!/usr/bin/env python
# coding: utf-8

# In[158]:


import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(cv2.__version__)


# # 고려 사항
#  1. json 파일 box 좌표가 어떤 걸 나타내는지 확인 (ai hub 데이터는 center_x, center_y, w, h 를 나타냄)
#  2. json 파일에 box 좌표가 몇개 있는지 확인
#  3. json 파일 내에 w, h, img_w, img_h 중 0값이 있는지 확인
#  3. class type 은 int로 변환 나머지 좌표값 및 w,h 는 float 형 유지

# # 데이터 경로 설정

# In[173]:


data_path = Path('D:\\Data\\YOLO\\Road_Obstacles\\2020-02-061.도로장애물표면인지(수도권)_sample')
label = data_path/ '061.도로장애물-표면인지영상(수도권)_샘플_라벨링데이터'
image = data_path/ '061.도로장애물-표면인지영상(수도권)_샘플_원천데이터'
txt = data_path/'061.도로장애물-표면인지영상(수도권)_샘플_txt데이터'


# # 데이터 확인

# In[174]:


data_json = label.glob('*.json') ## glob 함수는 generator을 반환해 주는 거여서 data_json을 계속해서 초기화 시켜야함
for j in data_json:
    with open(str(j)) as f:
        json_data = json.load(f)
        print(json.dumps(json_data ,indent = '\t')) ## '\t'를 통해 들여쓰기 설정


# # 이미지 라벨링 출력

# In[176]:


ex_img = cv2.imread(str(image / "V0F_HY_0093_20210129_133533_N_CH1_Seoul_Sun_Industrialroads_Day_07522.png"))
with open(str(label/'V0F_HY_0093_20210129_133533_N_CH1_Seoul_Sun_Industrialroads_Day_07522_BBOX.json')) as f:
    ex_label = json.load(f)
    box = ex_label['annotations'][0]['bbox']
print(box[0])

ff = np.fromfile('D:/Data/YOLO/Road_Obstacles/2020-02-061.도로장애물표면인지(수도권)_sample/061.도로장애물-표면인지영상(수도권)_샘플_원천데이터/V0F_HY_0093_20210129_133533_N_CH1_Seoul_Sun_Industrialroads_Day_07522.png', np.uint8)
ex_img = cv2.imdecode(ff,cv2.IMREAD_COLOR) ##imread 한글 경로가 오류남 fromfile ->imdecode 로 처리
print(np.shape(ex_img)[0])
cv2.rectangle(ex_img,(int(box[0]),int(box[1]),int(box[2]),int(box[3])),(0,0,255), 1) ## 
cv2.imshow('ex_img',ex_img)
cv2.waitKey()
cv2.destroyAllWindows()


# # 데이터 정제 
# # 1. dataframe 작성

# In[157]:


data_json = label.glob('*.json') ## glob 함수는 generator을 반환해 주는 거여서 data_json을 계속해서 초기화 시켜야함
df = pd.DataFrame({'file':[],
                       'class':[],
                       'center_x':[],
                       'center_y':[],
                       'width':[],
                       'height':[],
                       'img_width':[],
                       'img_height':[]})
box_size = []
for j in data_json:
    with open(str(j)) as f:
        json_data = json.load(f)
        box= json_data['annotations'][0]['bbox'] ## 이 데이터는 annotations를 list안에 dictionary 형태로 저장해둠 ->[0]을 통해 list 벗겨줌
        box_size.append(len(box))
        file_name = json_data['images']['file_name']
        class_num = json_data['annotations'][0]['category_id'] 
        center_x = box[0] 
        center_y = box[1]
        width = box[2]
        height = box[3]
        img_w=json_data['images']['width']
        img_h=json_data['images']['height']
        dic = {'file':file_name,'class':class_num,'center_x':center_x,'center_y':center_y,'width':width,'height':height,'img_width':img_w,'img_height':img_h}
        df = df.append(dic, ignore_index=True)
box_size.


# # 데이터 시각화

# In[151]:


fig, ax = plt.subplots(2,2,figsize = (15,15))
sns.boxplot(x = 'class', y='width',data=df, ax= ax[0][0])
sns.boxplot(x = 'class', y='height',data=df,ax= ax[0][1])
sns.boxplot(x = 'class', y='img_width',data=df, ax= ax[1][0])
sns.boxplot(x = 'class', y='img_height',data=df, ax= ax[1][1])
fig.tight_layout()
plt.show()


# # 2. 결측치 확인

# In[152]:


df.isnull().sum()


# # 3. 0 값 데이터 확인 및 삭제

# In[153]:


df_zero = df.copy()
list1 = ['class', 'width', 'height','img_width', 'img_height']

for i in list1:
    df_zero = df_zero.drop(df_zero[df_zero[i]==0].index)
    df_zero=  df_zero.reset_index(drop=True)
    

df_zero.describe()
df.describe()
    


# # 데이터 정제

# In[188]:


df_refine = df_zero.copy()

for i in df_refine.index:
    w=df_refine.loc[i,'img_width']
    h=df_refine.loc[i,'img_height']
    df_refine = df_refine.astype({'class':'int'})
    df_refine.loc[i,'center_x'] = df_refine.loc[i,'center_x']/w
    df_refine.loc[i,'width'] = df_refine.loc[i,'width']/w
    df_refine.loc[i,'center_y'] = df_refine.loc[i,'center_y']/h
    df_refine.loc[i,'height'] = df_refine.loc[i,'height']/h
    
df_refine[df_refine['file']=='V0F_HY_0093_20210129_133533_N_CH1_Seoul_Sun_Industrialroads_Day_07522.png']
    


# # dataframe txt 파일로 저장

# In[186]:


for i in df_refine.index:
    with open(txt/((df_refine.iloc[i,0][:-3])+'txt'), 'w') as f:
        for item in df_refine.iloc[i,1:6].tolist():
            f.write("%s " % item)
    

