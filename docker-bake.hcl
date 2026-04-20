variable "COMMIT_SHA" {
  default = "latest"
}

group "default" {
  targets = ["pocket-tts-server"]
}

target "pocket-tts-server" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/arm64", "linux/amd64"]
  tags = [
    "rg.fr-par.scw.cloud/namespace-unruffled-tereshkova/pocket-tts-server:${COMMIT_SHA}"
  ]
}
