#!/usr/bin/env bash
#
# Optexity Docker image build (use this `docker/` directory for all future container builds).
#
# VNC / browser-view work: latest flow lives on the `vnc` branch in optexity; build artifacts still
# ship from here (`docker/Dockerfile`, this script, supervisord, etc.).
#
# --- build.sh usage ---
#
#   ./build.sh --dev -t vnc --local
#
#   --dev    Target dev registry image: ghcr.io/optexity/opinference-dev (default without --dev is
#            ghcr.io/optexity/opinference).
#   -t, --tag <name>   Base image tag (default: latest). Platform is always appended, e.g. `-t vnc`
#            on arm64 -> .../opinference-dev:vnc-linux-arm64
#   --platform <os/arch>  Target platform (default: host native: linux/amd64 or linux/arm64).
#            Example: `--platform linux/amd64` on Apple Silicon for cross-builds.
#   --local  EC2 / air-gapped / no-GitHub: build and load into local Docker only — skips `gh` and
#            GHCR login, does not push. On machines with GitHub, omit --local to push to GHCR with
#            registry build cache.
#
# --- run (example: dev VNC image; tag includes platform, e.g. vnc-linux-arm64 on Apple Silicon) ---
#
# Do not commit real secrets; pass keys via env or an env-file.
#
#   sudo docker run \
#     -p 8080:8080 \
#     -p 9000:9000 \
#     --shm-size=2g \
#     -e USE_PLAYWRIGHT_BROWSER="False" \
#     -e GOOGLE_API_KEY="<set-me>" \
#     -e API_KEY="<set-me>" \
#     -e DEPLOYMENT=dev \
#     ghcr.io/optexity/opinference-dev:vnc-linux-arm64
#
# Exposed ports:
#   8080 — noVNC: open http://localhost:8080/vnc_lite.html?autoconnect=true&scale=true to view browsers
#   9000 — inference API: http://localhost:9000/inference (same as non-VNC deployments)
#

set -euo pipefail
set -x

readonly GHCR_REGISTRY="ghcr.io"
readonly GHCR_OWNER="optexity"
readonly IMAGE_PROD="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference"
readonly IMAGE_DEV="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference-dev"
readonly CACHE_REF="${GHCR_REGISTRY}/${GHCR_OWNER}/opinference-cache:buildcache"

TAG_DEV=0
LOCAL_MODE=0
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

detect_docker_platform() {
	case "$(uname -m)" in
		x86_64 | amd64)
			printf '%s' "linux/amd64"
			;;
		aarch64 | arm64)
			printf '%s' "linux/arm64"
			;;
		*)
			log "unsupported machine hardware name: $(uname -m); set --platform explicitly" >&2
			return 1
			;;
	esac
}

platform_tag_suffix() {
	local plat="$1"
	printf '%s' "${plat//\//-}"
}

is_linux() {
	[[ "$(uname -s)" == "Linux" ]]
}

log() {
	printf "[build.sh] %s\n" "$*" >&2
}

ensure_dependencies() {
	local missing=()
	if is_linux; then
		for cmd in docker; do
			command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
		done
		if [[ "$LOCAL_MODE" -ne 1 ]]; then
			command -v gh >/dev/null 2>&1 || missing+=("gh")
		fi
	else
		for cmd in colima docker; do
			command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
		done
		if [[ "$LOCAL_MODE" -ne 1 ]]; then
			command -v gh >/dev/null 2>&1 || missing+=("gh")
		fi
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

configure_docker_env() {
	export DOCKER_BUILDKIT=1
	if is_linux; then
		log "linux: using default docker socket (not colima)"
		return 0
	fi
	export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
}

ensure_gh_authenticated() {
	# GH_TOKEN / GITHUB_TOKEN are honoured natively by gh CLI — no login needed.
	if [[ -n "${GH_TOKEN:-}" || -n "${GITHUB_TOKEN:-}" ]]; then
		log "gh CLI: using GH_TOKEN / GITHUB_TOKEN env var"
		return 0
	fi

	if gh api user >/dev/null 2>&1; then
		log "gh CLI already authenticated"
		return 0
	fi

	log "gh CLI not authenticated or token invalid; launching login"
	log "  (headless/EC2: set GH_TOKEN=<pat> or GHCR_TOKEN=<pat> GHCR_USERNAME=<user> to skip browser auth)"
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

ensure_buildx_builder() {
	local builder="optexity-builder"
	if docker buildx inspect "${builder}" >/dev/null 2>&1; then
		docker buildx use "${builder}"
		log "buildx: using existing builder '${builder}'"
	else
		log "buildx: creating docker-container builder '${builder}' (required for registry cache)"
		docker buildx create --name "${builder}" --driver docker-container --use --bootstrap
	fi
}

start() {
	cd_to_script_dir
	if ! is_linux; then
		ensure_colima_running
	fi
	configure_docker_env
	if [[ "$LOCAL_MODE" -ne 1 ]]; then
		ensure_buildx_builder
	fi
}

login() {
	docker_ghcr_login
}

build() {
	local image_ref="" tag_suffix platform_tag
	tag_suffix="$(platform_tag_suffix "${DOCKER_PLATFORM}")"
	platform_tag="${IMAGE_TAG}-${tag_suffix}"
	if [[ "$TAG_DEV" -eq 1 ]]; then
		image_ref="${IMAGE_DEV}:${platform_tag}"
	else
		image_ref="${IMAGE_PROD}:${platform_tag}"
	fi

	log "platform=${DOCKER_PLATFORM} image=${image_ref}"

	if [[ "$LOCAL_MODE" -eq 1 ]]; then
		log "local mode: building image into Docker (no GHCR login or push)"
		docker buildx build \
			--build-arg CACHE_BREAK=$(date +%s) \
			--platform="${DOCKER_PLATFORM}" \
			-t "${image_ref}" \
			--load .
	else
		docker buildx build \
			--build-arg CACHE_BREAK=$(date +%s) \
			--platform="${DOCKER_PLATFORM}" \
			--cache-from=type=registry,ref="${CACHE_REF}" \
			--cache-to=type=registry,ref="${CACHE_REF}",mode=max \
			-t "${image_ref}" \
			--push .
	fi
}

main() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--local)
				LOCAL_MODE=1
				shift
				;;
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
			--platform)
				if [[ -z "${2:-}" ]]; then
					log "error: $1 requires a value (e.g. $1 linux/amd64)" >&2
					exit 1
				fi
				DOCKER_PLATFORM="$2"
				shift 2
				;;
			*)
				log "unknown argument: $1 (supported: --local, --dev, --tag|-t <tag>, --platform <os/arch>)" >&2
				exit 1
				;;
		esac
	done

	if [[ -z "${DOCKER_PLATFORM}" ]]; then
		DOCKER_PLATFORM="$(detect_docker_platform)" || exit 1
	fi

	ensure_dependencies
	start
	if [[ "$LOCAL_MODE" -ne 1 ]]; then
		login
	fi
	build
}

main "$@"
