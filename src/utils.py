import os
import json

####################################################################################################
# 1. json파일에서 카테고리 매핑을 만드는 함수
def get_category_mapping(ann_dir, debug=False, add_more=False, return_types=None):
    """
    어노테이션 디렉토리 내 JSON 파일들을 탐색하여
    ID, 이름, 인덱스 간 매핑을 생성하고 요청된 매핑을 반환하는 함수입니다.

    Args:
        ann_dir (str): 어노테이션 JSON 파일들이 저장된 디렉토리 경로
        debug (bool): 디버깅 출력 여부
        add_more (bool): True인 경우 'Background'와 'No Class'를 포함하여 인덱싱  
                        False인 경우 0부터 시작하는 인덱스만 부여
        return_types (Optional[List[str]]):  
            반환할 매핑 키 리스트 (선택 가능한 매핑: 'id_to_name', 'name_to_id',  
            'name_to_idx', 'idx_to_name', 'id_to_idx', 'idx_to_id')

    Returns:
        Dict[str, Any]: 요청된 매핑 딕셔너리

    Note:
        - 중복된 카테고리 이름은 제거 후 정렬합니다.
        - add_more=True인 경우: 0은 'Background', 마지막 인덱스는 'No Class'로 지정
        - add_more=False인 경우: 인덱스는 0부터 순차 부여
        - 반환 매핑은 학습 및 시각화 등에 활용됩니다.
    """
    # 디버깅 메시지 출력
    if not isinstance(ann_dir, str):
        raise TypeError(f"ann_dir는 문자열(str)이어야 합니다. 현재 타입: {type(ann_dir)}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"ann_dir 경로가 존재하지 않습니다: {ann_dir}")
    if not isinstance(add_more, bool):
        raise TypeError(f"add_more는 불리언 값이어야 합니다. 현재 타입: {type(add_more)}")
    
    id_to_name = {}
    # 어노테이션 폴더 내 파일 순회
    for file in os.listdir(ann_dir):
        file_path = os.path.join(ann_dir, file)

        if file.endswith(".json"):
            # JSON 파일 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
                categories = ann.get('categories', [])

                for cat in categories:
                    id_to_name[cat['id']] = cat['name']

    # 이름 기준 정렬
    sorted_names = sorted(set(id_to_name.values())) # 중복 제거 후 정렬

    name_to_idx = {}
    if add_more:
        # 1부터 인덱싱, 0은 배경, 마지막 숫자는 No Class
        name_to_idx = {'Background': 0}
        for idx, name in enumerate(sorted_names, start=1):
            name_to_idx[name] = idx
        name_to_idx['No Class'] = len(name_to_idx)
    else:
        for idx, name in enumerate(sorted_names):
            name_to_idx[name] = idx

    # 역 매핑
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    name_to_id = {name: id for id, name in id_to_name.items()}

    # 아이디 <-> 인덱스
    id_to_idx = {id_: name_to_idx[name] for id_, name in id_to_name.items()}
    idx_to_id = {idx: name_to_id[name] for idx, name in idx_to_name.items() if name in name_to_id}
    
    # 모든 매핑 저장
    all_mappings = {
        'id_to_name': id_to_name,
        'name_to_id': name_to_id,
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name,
        'id_to_idx': id_to_idx,
        'idx_to_id': idx_to_id
    }

    if return_types is None:
        result = all_mappings
    elif len(return_types) == 1:
        result = all_mappings[return_types[0]]
    else:
        result = tuple(all_mappings[k] for k in return_types)
    
    # if debug:
    #     print(f"[DEBUG] 반환 직전 매핑 결과:")
    #     if isinstance(result, dict):
    #         for key, value in result.items():
    #             print(f"  {key}: {value}")
    #     elif isinstance(result, tuple):
    #         for idx, item in enumerate(result):
    #             print(f"  반환 항목 {idx+1}: {item}")
    #     else:
    #         print(f"  반환 값: {result}")

    return result