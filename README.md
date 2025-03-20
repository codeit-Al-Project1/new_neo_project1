# Git stash -- push/ pull 진행 상황 저장 및 업데이트 방지
 
 Git pull을 할 경우 main의 혹은 자신의 branch의 데이터를 가져와 merge를 한다.
 이경우 현재 진행상황 및 등을 덮어쓰게 되는게 안전하게 보호 하는 방법은 git stash를 이용하여
 보관하는 방법이 있다.
 
 git add <파일/폴더 명>
 
 git add로 어떤 파일을 임시저장할지 지정한다.
 
 보통의 경우 다음단계로 commit을 한다면, 임시저장은 stash로 진행하며
 
 git stash -m "임시저장"
 
 commit과 비슷한 형태로 임시저장한다.
 
 다음 pull을 하여 자신의 디랙토리가 최신화시킨 후 push를 진행한다.
 
 그 다음, 바뀐 부분 혹은 진행상황을 돌리기 위해 stash pop을 이용하여
 
 진행상황으로 되돌린다.