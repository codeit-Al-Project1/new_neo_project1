import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import gdown
import shutil
from tqdm import tqdm
import json

def download_data(download_path = './data', download=True, extract=True):
    """
    Kaggle API를 이용해 데이터를 다운로드하고, 압축을 해제하는 메소드, 
    메소드의 마지막 과정에서 압축파일을 제거한다.

    ***Note: Kaggle API를 사용하기 위해서는 Kaggle 계정이 있어야 하며,
    Kaggle API Token이 필요하다.***
    ***Note: 다운로드 위치를 주의하도록 하자.***
    
    :param download_path: str: 다운로드 경로
    :param download: bool: 데이터 다운로드 여부
    :param extract: bool: 압축 해제 여부
    """
    if os.path.exists(download_path):
        download_path = './data'
        print("data폴더가 존재합니다.")
    elif download_path != './data' and not os.path.exists(download_path):
        os.mkdir(download_path)
        print("지정한 위치에 새로운 폴더를 생성합니다.")
    else:
        download_path = './'
        print("data폴더가 없습니다. 폴더를 새로 만들지 않고 현재폴더에서 진행됩니다.")

    competition_name = 'ai01-level1-project'

    if download:
        api = KaggleApi()
        api.authenticate()

        if f"{competition_name}.zip" in os.listdir(download_path):
            print(f"{competition_name}가 {download_path}에 이미 존재합니다.")
        else:
            print(f"{competition_name} 다운로드 시작...")
            try:
                api.competition_download_files(competition_name, path=download_path, quiet=False)
                print(f"다운로드 완료! 저장위치: {download_path}")
            except Exception as e:
                print(f"Error: {e}")
                return           
            
    if extract:
        zip_file = f'{competition_name}.zip'
        if zip_file not in os.listdir(download_path):
            print(f"{zip_file}를 {download_path}에서 찾을 수 없습니다.")
            return
        
        zip_path = os.path.join(download_path, zip_file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
            print(f"{zip_file} 압축 해제 완료!")
        
        os.remove(zip_path)
        print(f"{zip_file} 파일 삭제 완료!")
    
    return

def wrap_annotation(path = './data'):
    """
    수정된 train_annotations 압축 파일을 내려받아,
    기존 train_annotations 폴더를 대체 하는 과정.
    """

    if os.path.exists(path):
        path = './data'
    else:
        path = './'

    zip_name = "train_annotations"
    zip_path = os.path.join(path, f"{zip_name}.zip")

    file_id = "1dCMvEIqIhbJKa8G5poO5MPdeDHTiEcQ8"

    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    print("Train_annotations를 덮어씌웁니다.")
    gdown.download(url, os.path.join(path, f'{zip_name}.zip'), quiet=False)

    if os.path.exists(os.path.join(path, zip_name)):
        shutil.rmtree(os.path.join(path, zip_name))
        print("기존 데이터 삭제 후 대체합니다.")

    # 압축 해제
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        print("Train_annotations 압축 해제 완료!")
        os.remove(zip_path)  # 다운로드한 zip 파일 삭제
        print(f"{zip_name}.zip 파일 삭제 완료!")
    else:
        print(f"{zip_path} 파일을 찾을 수 없습니다.")
    
def folder_check():
    if os.path.exists('./data'):
        path = './data'
    else:
        path = './'

    annot_folder = "train_annotations"

    annot_path = os.path.join(path, annot_folder)
    
    count = 0

    # 모든 파일을 먼저 수집하고 그 후 tqdm으로 진행 표시
    all_files = []
    for root, dirs, files in os.walk(annot_path):
        for file in files:
            if file.endswith('.json'):
                all_files.append(os.path.join(root, file))

    # 파일이 존재하면 tqdm으로 진행 상태 표시
    if all_files:
        for json in tqdm(all_files, desc="Counting JSON files", unit="file", mininterval=0.5, maxinterval=2):
            count += 1
        print(f"{count}개가 폴더 내부에 있습니다.")
    else:
        print("폴더 내부에 파일이 없습니다.")
        
    return


if __name__ == '__main__':
    download_data(download=False, extract=False)
    folder_check()
    wrap_annotation()
    folder_check()