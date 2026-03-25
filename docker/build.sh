#!/usr/bin/env bash

set -euo pipefail
set -x

readonly GHCR_REGISTRY="ghcr.io"
readonly GHCR_OWNER="optexity"
readonly IMAGE_PROD="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference"
readonly IMAGE_DEV="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference-dev"
readonly CACHE_REF="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference-cache:buildcache"

TAG_DEV=0
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

is_linux() {
	[[ "$(uname -s)" == "Linux" ]]
}

log() {
	printf "[build.sh] %s\n" "$*" >&2
}

ensure_dependencies() {
	local missing=()
	if is_linux; then
		for cmd in docker gh; do
			command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
		done
	else
		for cmd in colima docker gh; do
			command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
		done
	fi
	if ! docker buildx version >/dev/null 2>&1; then
		missing+=("docker-buildx")
	fi

	if [[ ${#missing[@]} -gt 0 ]]; then
		log "missing dependencies: ${missing[*]}; running install.sh"
		bash "${SCRIPT_DIR}/install.sh"
	fi
}

cd_to_script_dir() {
	cd "$SCRIPT_DIR"
}

ensure_colima_running() {
	if colima status 2>/dev/null | grep -q "Running"; then
		log "colima already running; skipping start"
		return 0
	fi

	log "starting colima"
	colima start --cpu 4 --memory 8 --disk 100
}

ensure_ssh_agent_has_key() {
	if ssh-add -l >/dev/null 2>&1; then
		log "ssh-agent already has keys"
		return 0
	fi

	log "starting ssh-agent and adding default key"
	eval "$(ssh-agent -s)"
	ssh-add "${HOME}/.ssh/id_rsa"
	ssh-add -l
}

configure_docker_env() {
	export DOCKER_BUILDKIT=1
	if is_linux; then
		log "linux: using default docker socket (not colima)"
		return 0
	fi
	export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
}

ensure_gh_authenticated() {
	if gh auth status >/dev/null 2>&1; then
		log "gh CLI already authenticated"
		return 0
	fi

	log "gh CLI not authenticated; launching login"
	gh auth login --hostname github.com --git-protocol https --scopes write:packages,read:packages
}

docker_ghcr_login() {
	local token username
	if [[ -n "${GHCR_TOKEN:-}" ]]; then
		token="$GHCR_TOKEN"
		username="${GHCR_USERNAME:-token}"
	else
		ensure_gh_authenticated
		token="$(gh auth token)"
		username="$(gh api user --jq .login)"
	fi

	log "logging docker into GHCR"
	echo "${token}" | docker login "${GHCR_REGISTRY}" --username "${username}" --password-stdin
}

start() {
	cd_to_script_dir
	if ! is_linux; then
		ensure_colima_running
	fi
	ensure_ssh_agent_has_key
	configure_docker_env
}

login() {
	docker_ghcr_login
}

build() {
	local image_tag=""
	if [[ "$TAG_DEV" -eq 1 ]]; then
		image_tag="${IMAGE_DEV}:${IMAGE_TAG}"
	else
		image_tag="${IMAGE_PROD}:${IMAGE_TAG}"
	fi

	docker buildx build \
		--platform=linux/amd64 \
		--cache-from=type=registry,ref="${CACHE_REF}" \
		--cache-to=type=registry,ref="${CACHE_REF}",mode=max \
		--ssh default \
		-t "${image_tag}" \
		--push .
}

main() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--dev)
				TAG_DEV=1
				shift
				;;
			--tag|-t)
				if [[ -z "${2:-}" ]]; then
					log "error: $1 requires a tag value (e.g. $1 v1.2.3)" >&2
					exit 1
				fi
				IMAGE_TAG="$2"
				shift 2
				;;
			*)
				log "unknown argument: $1 (supported: --dev, --tag|-t <tag>)" >&2
				exit 1
				;;
		esac
	done

	ensure_dependencies
	start
	login
	build
}

main "$@"
