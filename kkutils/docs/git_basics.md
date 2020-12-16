# Git Basics
The document outlines all of the basic git commands that you use the most often.

## Clone
This command clones a repository to your local system.

```shell
git clone REPO_URL
```

Example:
```shell
https://github.com/PT-ML/opticalflow.git
```
![](https://i.imgur.com/LziL2Um.png)


## Branch
Branch shows you all existing branches.
```shell
git branch -a
```
![](https://i.imgur.com/TR23s1R.png)
The * symbol shows which branch you are currently on.

## Checkout
Checkout lets you move between existing branches.
```shell
git checkout BRANCH_NAME
```
![](https://i.imgur.com/BKcknmY.png)


Checkout also lets you move between existing commits.
```shell
git checkout COMMIT_NUMBER
```
![](https://i.imgur.com/BpDNyrF.png)

You can create a new branch with checkout as well.
```shell
git checkout -b NEW_BRANCH
```
![](https://i.imgur.com/VGcs9X5.png)


## Add
Add takes all of the new changes that you've made to the repository and stages them for a commit.

You can add files one at a time.
```shell
git add FILEPATH0 FILEPATH1 FILEPATH2 ...
```

You can also add all modified files at once.
```shell
git add -a
```

![](https://i.imgur.com/GyfiqW0.png)
![](https://i.imgur.com/TG4pGXr.png)
![](https://i.imgur.com/Z04FdCo.png)

## Reset
The reset command resets all files that you have staged for commit. The contents of the files stay the same.
```shell
git reset
```
![](https://i.imgur.com/xGlvKhF.png)
![](https://i.imgur.com/S5nijt3.png)


Using the --hard parameter reverts the contents of the files that you changed.
```shell
git reset --hard
```

## Commit
Commit takes all of the staged changes that you added and finalizes them with a message.

```shell
git commit -m "
Write an explanation of what you changed here.
"
```
![](https://i.imgur.com/TobQP6N.png)


## Status
The status command shows you the status of all modified files. Files displayed in red haven't been staged for commit. Files displayed in green are staged for commit.

```shell
git status
```
![](https://i.imgur.com/TG4pGXr.png)

## Push
The push command uploads your commits to github.

When you are pushing a commit to a new branch, you need to make the new branch upstream to origin.
```shell
git push -u origin NEW_BRANCH
```
![](https://i.imgur.com/qCkUSFB.png)

Otherwise, when you are pushing commits to an existing branch, you can just type in the following:
```shell
git push
```
![](https://i.imgur.com/KzCPgMm.png)


## Pull
The pull command downloads all of the changes pushed to github to your local system.
```shell
git pull
```
![](https://i.imgur.com/UEikrwa.png)

## Log
The log command shows you all of the commits that load up to your current commit number.

```shell
git log
```
![](https://i.imgur.com/R0Kc7bh.png)

Note: If you want to checkout a specific commit number, you can get the commit number from the log.

Using the --stat method shows you commit statistics.
![](https://i.imgur.com/D3B0nx9.png)

## Pull Requests
In order to merge a branch into another branch, for example the development branch, you will need to send a pull request to whoever is in charge of the development branch.
![](https://i.imgur.com/6bVucTu.png)
![](https://i.imgur.com/9reLZtA.png)
Once the pull request is made, it can be reviewed by another team member and merged.
![](https://i.imgur.com/8nDLVwG.png)
After the 'Confirm' button is pressed, the pull request will be closed and the corresponding branches will be merged.
![](https://i.imgur.com/0wxoLuH.png)

## Looking At Commits And Merges on Github
![](https://i.imgur.com/EXROkSk.png)
If you change to a given branch and click on the commits button, you can see a list of commits.
![](https://i.imgur.com/iIvS4Pc.png)
Clicking on one of the commits shows you details on what was changed in each file during that commit.
![](https://i.imgur.com/hfBNRnu.png)

## The Development Branch Is Ahead Of My Working Branch. What should I do?
There are going to be times when the development branch is updated, and those updates aren't included in your current working branch. Let's also say that you made changes to your local branch.
In this case, you will need to merge the development branch into your local branch.

First, make sure that you pull the latest changes from github.
![](https://i.imgur.com/a2oVXEB.png)
Then merge the development branch into your local branch.
![](https://i.imgur.com/HbBRTVB.png)
The output messages will tell you which files are in conflict.
You will resolve each of these conflicts before your next commit.
Note that checking the git status at this time will also show you which files are in conflict.
![](https://i.imgur.com/IgLsMFO.png)
In this example, docs/git_basics.md on the cm107 branch is in conflict with docs/git_basics.md on the development branch.
If you open docs/git_basics.md with VSCode, you'll see the following.
![](https://i.imgur.com/ybvLqAe.png)
The current change and the incoming changes are both shown.
In order to choose changes you want to keep, you can press one of the following buttons.
![](https://i.imgur.com/9RkU2Zz.png)
Pressing the 'Accept Incoming Change' button gives you the following result.
![](https://i.imgur.com/bJrANqz.png)
After you have fixed all merge conflicts, you can add all modified files to stage them for commit, make a commit, and then push your changes.
![](https://i.imgur.com/cIj0Hg3.png)
![](https://i.imgur.com/DfHwRYq.png)

## Deleting Unnecessary Files/Directories In Your Repository
Deleting a file locally in your repository will not delete the file on github.
When you want to remove a file from the github repository, you must use the git rm command.

```shell
git rm FILE_PATH
```
![](https://i.imgur.com/buCI9C9.png)

In the case of a directory, you will need to use the -rf parameter to recursively delete every file in the directory.

```shell
git rm -rf DIR_PATH
```