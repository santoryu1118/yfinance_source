# kafka : sh start_docker.sh kafka
# spark : sh start_docker.sh spark

docker-compose -f $PWD/$1-docker-compose.yaml down
docker-compose -f $PWD/$1-docker-compose.yaml up -d --build