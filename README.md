 # Data Download 실행
 필요한 Data Download 및 extract 그리고 wrapping과정을 통합한 모듈
 ***NOTE: colab과의 연동을 진행한 vs code에서는 다운로드 진행상황이 보이나, 일반 vs code환경에서의 실행시 진행상황은 보이지 않습니다. (1분에서 5분 소요)***
 ***Kaggle API를 이용한 데이터 다운로드 이므로, Kaggle 인증이 필요합니다. 다운로드를 따로하려면 아래의 커멘드 참조***
 실행: -- 수정
  python src/data_utils/data_download.py --download True --extract True
  
  다운로드 = False 압축해제 = False --> 압축파일을 가지고있고 모듈을 통해 해제하려면 './data'에 압축파일을 두고 extract만 True로 진행.
  
  python src/data_utils/data_download.py --download False --extract False