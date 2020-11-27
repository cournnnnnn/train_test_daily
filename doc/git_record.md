terminnal/cmd 基本操作
    1.本地初始化一个文件夹做为仓库
        git init
    2.添加文件到head里面
        git add filename
    3.添加操作说明，并提交
        git commit -m ’操作说明‘
    4.上传到远程仓库
        git push origin master
注释：origin为远程仓库别名，master为分支名
git remote add origin git@github.com:yourName/yourRepo.git

服务器克隆到本地
    git clone 远程地址
本地到远程服务器
    git push origin master
    
git push 前最好 git pull 一下

遇到问题：
    q1：push 更新被拒绝，因为您当前分支的最新提交落后于其对应的远程分支。
    a1：先pull一下，不行再   
    q2：git pull origin master 拒绝合并无关的历史的错误解决
    a2：git pull origin master --allow-unrelated-histories
    q3：git push 遇到文件冲突
    a3：按提示rm/add后，需要commit才算完成冲突处理
    
    
1.git的撤销操作

撤销操作

        git status 首先看一下add 的文件
        git reset HEAD  上一次add 里面的全部撤销了
        git reset HEAD fileName  对某个文件进行撤销了

2. git commit 错误
         git add后 ， 又 git commit 了。
首先
         git log 查看节点
         commit  YYYYYYYYYYYYYYYYYYYYY  
然后
        git reset commit_id
        commit_id 可以是简写前几位
还没有 push  的时候
         git reset commit_id （回退到上一个 提交的节点 代码还是原来自己修改的）
         git reset –hard commit_id （回退到上一个commit节点， 代码也发生了改变，变成上一次的，本次的修改也丢了）

3.如果是push了以后，可以使用 git revert

还原已经提交的修改 ，此次操作之前和之后的commit和history都会保留，并且把这次撤销作为一次最新的提交
git revert HEAD 撤销前一次 commit
git revert HEAD^ 撤销前前一次 commit
git revert commit-id (撤销指定的版本，撤销也会作为一次提交进行保存）
git revert 是提交一个新的版本，将需要revert的版本的内容再反向修改回去，版本会递增，不影响之前提交的内容。

