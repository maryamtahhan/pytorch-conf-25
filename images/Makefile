# Makefile to build and push container images

IMAGE_REGISTRY=quay.io/mtahhan
LATEST_TAG=vllm-latest
IMAGE_NAME=pytorch-conf-demo
IMAGE=$(IMAGE_REGISTRY)/$(IMAGE_NAME)
OLD_TAG=vllm-old
CONTAINER_TOOL = $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)

.PHONY: all build-latest build-old push-latest push-old

all: build-latest build-old

build-latest:
	@$(CONTAINER_TOOL) build --build-arg ENTRYPOINT_PATH=$(LATEST_TAG) -t $(IMAGE):$(LATEST_TAG) -f $(LATEST_TAG)/Containerfile.vllm-rocm .

build-old:
	@$(CONTAINER_TOOL) build --build-arg ENTRYPOINT_PATH=$(OLD_TAG) -t $(IMAGE):vllm_nightly_main_20250212 -f $(OLD_TAG)/Containerfile.vllm-rocm-old .

push-latest:
	@$(CONTAINER_TOOL) push $(IMAGE_REGISTRY):$(LATEST_TAG)

push-old:
	@$(CONTAINER_TOOL) push $(IMAGE_REGISTRY):$(OLD_TAG)