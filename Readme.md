## Recommender System 
python 으로 matrix factorization 구현

 ### Dataset
 - MovieLens 20M Dataset
 - 파일 정보
   - 크기 : 190 MB
   - 라인 수 : 20,000,263
 - 라벨 정보
   - Rating : 데이터 세트의 3 번째 컬럼
 - 학습 데이터
   - 날짜 기준으로 잘라서 사용
   - Dataset1
      - 2005-01-01 (timestamp >= 1104505203) ~ 2008-12-31 (timestamp <= 1230735592)
      - 총 데이터 크기(개수) : 5,187,587
   - Dataset2
      - 2013-12-31 (including 2013-12-31, timestamp < 1388502017) 이전의 데이터
      - 총 데이터 크기(개수) : 
 - 평가 데이터
   - Dataset1
      - 2009-01-01 (timestamp >= 1230735600) ~ 2009-12-31 (timestamp <= 1262271552)
      - 총 데이터 크기(개수) : 930,093
   - Dataset2
      - 2014-01-01 (including 2014-01-01, timestamp >= 1388502017) 이후의 데이터
      - 총 데이터 크기(개수) : 
   
  ### 파일 설명
 - ratings.csv : 입력 데이터 파일 
 - Dataset1/
     - train.py : 모델 학습/테스트 진행(main)
     - model.py : 모델 구현
     - data_provider.py : 데이터 로드/저장     
     - B_results_DS1.csv : Dataset1에 대한 결과 파일
 - Dataset2/
     - train.py : 모델 학습/테스트 진행(main)
     - model.py : 모델 구현
     - data_provider.py : 데이터 로드/저장     
     - B_results_DS2.csv : Dataset2에 대한 결과 파일  
 
 ### 실행 방법 
 - 각 폴더(DataSet1, Dataset2)내의 train.py 실행 

 ### 모델 구조
 - *Matrix Factorization Techniques for Recommender Systems* 논문에 제시된 모델 중 bias를 추가한 모델
 - Input : UserId, MovieId
 - Output : rating
 - Learning Parameters
    - user latent feature P
    - movie latent feature Q
    - user bias b_u
    - movie bias b_i        
 - global bias b 는 학습데이터 내 rating 값들의 평균 값(고정) 
 - 평가지표 : Rooted Mean Square Error
    
 ### Hyperparameter    
 - k = 5
 - epoch 
    - DS1 model = 30
    - DS2 model = 10
 - learning_rate = 0.01
 - regularization parameter = 0.01
    
 ### 학습 시간(cpu만 사용)
 - DS1 model : 82min 33.7943secs
 - DS2 model : 2h 28m 16.6724s
 
 ### 학습 결과(train/test data RMSE value)
 - DS1 model
    - train data : 0.7559
    - test data  : 1.1386
 - DS2 model
    - train data : 0.8561
    - test data  : 1.1028
