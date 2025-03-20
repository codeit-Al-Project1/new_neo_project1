 # Data Download 실행

<<<<<<< Updated upstream
 python src/data_utils/data_download.py --download True --extract True

 Download path는 default 값인 './data'로 설정 된다. 필요하다면 아래의 방식으로 재설정 하는 방식이 가능하되, 가능하면 defalut값으로 진행하는 것을 권장한다.

 python src/data_utils/data_download.py --download_path <다운로드 경로> --download True --extract True

 ***Note: 다운로드 과정에 있어서 kaggle의 API를 적용했으므로, kaggle인증을 미리 진행해야 다운로드를 진행할 수 있다.***

 ***Note: 다운로드 과정이 필요없는 경우, train_annotations폴더 내의 annotations의 최신화를 위해 download False, extract False를 두어 실행한다.***

  python src/data_utils/data_download.py --download False --extract False

# json을 모델에 학습시키기 위해 수정하는 작업

 실행 방법:

  python src/data_utils/data_loader.py

  ***NOTE: Defalut 값으로 './data/train_annotations'를 추적하며 './data/train_annots_modify' 폴더를 생성한다.***

  ***NOTE: argument parse 값으로 output dir와 json_folder설정이 가능하나, 추후 다른 모듈과의 연동을 위해 기본값으로 진행하는것을 권장한다.***
=======
진행상황을 업데이트 한다.
>>>>>>> Stashed changes
