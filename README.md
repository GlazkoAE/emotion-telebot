# emotion-telebot
Demonstration service for ITMO computer vision project

Run docker:
```
cd path/to/project/repo
docker build -t emotion-telebot .
docker run --name <container name> -d -e key=<secret telegram token> emotion-telebot
```