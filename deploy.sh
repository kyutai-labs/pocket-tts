set -ex
COMMIT_SHA=$(git rev-parse --short HEAD)
docker compose -f swarm-config.yaml build  --push