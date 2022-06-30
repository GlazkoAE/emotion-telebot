# emotion-telebot
Demonstration service for ITMO computer vision project

Download models from [google drive](https://drive.google.com/drive/folders/1k3lgX4HtS73vSr99TwhIs-zerVzDxANL?usp=sharing)
and save them into `<project_root>/saved_models/`

Run docker:
```
cd path/to/project/repo
docker build -t emotion-telebot .
docker run --name <container name> -d -e key=<secret telegram token> emotion-telebot
```
