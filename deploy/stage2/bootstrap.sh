#!/usr/bin/env bash
set -euo pipefail
umask 027

mode=""
remaining_hours=""
while (($#)); do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing hours}"; shift 2 ;;
    *) echo "unknown bootstrap argument: $1" >&2; exit 64 ;;
  esac
done
[[ "$mode" == "preflight" ]] || { echo "--mode preflight is required" >&2; exit 64; }
[[ "$remaining_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "valid remaining rental hours required" >&2; exit 64; }

cd /opt/tarca
mkdir -p artifacts/stage2/runtime
python scripts/run_stage2_v1.py prepare \
  --repository-root /opt/tarca --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2
python - "$remaining_hours" <<'PY'
import hashlib, json, os, pathlib, shutil, sys, tempfile
import psutil, torch, yaml

hours = float(sys.argv[1])
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.2.2"
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert psutil.cpu_count(logical=False) >= 28
assert psutil.virtual_memory().total >= 224 * 1024**3
assert shutil.disk_usage("artifacts").free >= 200 * 1024**3
assert hours > 1.0
for index in range(2):
    props = torch.cuda.get_device_properties(index)
    assert "4090" in props.name and props.total_memory >= 24 * 1024**3
config = yaml.safe_load(pathlib.Path("configs/stage2/stage2_v1.yaml").read_text())
for source in config["sources"]:
    root = pathlib.Path("third_party/stage2") / source["source_id"]
    for asset in source["assets"]:
        path = root / asset["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
x = torch.linspace(-1, 1, 4096, device="cuda:0")
fp32 = (x.float() * x.float()).sum()
with torch.autocast("cuda", dtype=torch.float16):
    amp = (x * x).sum()
assert torch.isfinite(fp32) and torch.isfinite(amp)
with tempfile.NamedTemporaryFile(dir="artifacts/stage2/runtime", delete=False) as handle:
    checkpoint = pathlib.Path(handle.name)
torch.save({"probe": x[:16].cpu()}, checkpoint)
loaded = torch.load(checkpoint, map_location="cpu")
checkpoint.unlink()
assert torch.equal(loaded["probe"], x[:16].cpu())
evidence = {
    "status": "PREFLIGHT_PASS",
    "remaining_rental_hours": hours,
    "gpu_count": 2,
    "source_hashes_verified": True,
    "checkpoint_roundtrip_passed": True,
    "formal_tasks_executed": 0,
}
pathlib.Path("artifacts/stage2/runtime/bootstrap_evidence.json").write_text(
    json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
)
PY
python scripts/run_stage2_v1.py preflight \
  --repository-root /opt/tarca --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2 \
  --evidence artifacts/stage2/runtime/bootstrap_evidence.json
echo "PREFLIGHT_PASS: no training or formal task was started"
