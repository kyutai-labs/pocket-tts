set -ex
export COMMIT_SHA=$(git rev-parse --short HEAD)
docker buildx bake -f docker-bake.hcl --push

docker -H ssh://root@pocket-tts-api.kyutai.org stack deploy \
    --with-registry-auth -c swarm-config.yaml pocket-tts
