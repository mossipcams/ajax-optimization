#!/usr/bin/env bash
set -euo pipefail

vm_repo="${SAYSO_VM_REPO:-LLM@192.168.1.76:ajax-optimization}"

git fetch --no-tags "$vm_repo" '+refs/sayso/*:refs/sayso/*'
git push origin 'refs/sayso/*:refs/sayso/*'
