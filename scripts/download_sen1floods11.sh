#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/data/raw/Sen1Floods11"
BUCKET="gs://sen1floods11"

mkdir -p "${OUTPUT_DIR}"

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is not installed or not available in PATH."
  echo "Install Google Cloud SDK first:"
  echo "  https://cloud.google.com/sdk/docs/install"
  echo ""
  echo "After installation, inspect the bucket with:"
  echo "  gsutil ls ${BUCKET}"
  echo ""
  echo "Then download with:"
  echo "  gsutil -m rsync -r ${BUCKET} data/raw/Sen1Floods11"
  exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
  echo "Listing Sen1Floods11 bucket:"
  gsutil ls "${BUCKET}"
  exit 0
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Source bucket: ${BUCKET}"
echo ""
echo "This command uses rsync without deletion flags, so local extra files are not removed."
echo "Starting download/synchronization..."

cd "${PROJECT_ROOT}"
gsutil -m rsync -r "${BUCKET}" "data/raw/Sen1Floods11"

echo "Sen1Floods11 synchronization complete."
