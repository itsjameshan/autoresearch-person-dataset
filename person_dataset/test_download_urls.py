"""Reachability test for YOLOv12 model/dataset download URLs.

Probes each URL with a Range: bytes=0-8191 GET request to verify without
downloading the full file. Classifies each result as OK / Landing / Failed.
"""
import ssl
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

ssl._create_default_https_context = ssl._create_unverified_context

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
TIMEOUT = 15

# Format: (category, description, url, expected_type)
# expected_type: "binary" -> should return a file, "page" -> HTML is OK
URLS = [
    # === Set A: repo model mirrors (yolo12s.pt) ===
    ("repo-mirror", "gitmirror.com yolo12s.pt",
     "https://hub.gitmirror.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
     "binary"),
    ("repo-mirror", "gh.api.99988866.xyz yolo12s.pt",
     "https://gh.api.99988866.xyz/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
     "binary"),
    ("repo-mirror", "fastgit.org yolo12s.pt",
     "https://download.fastgit.org/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
     "binary"),
    ("repo-mirror", "github.com yolo12s.pt (official)",
     "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
     "binary"),

    # === Set B: doubao thread - direct-file dataset URLs ===
    ("dataset-file", "UCF-QNRF zip",
     "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_Crowd_Dataset.zip",
     "binary"),
    ("dataset-file", "NWPU-Crowd zip",
     "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/NWPU-Crowd.zip",
     "binary"),
    ("dataset-file", "JHU-CROWD++ zip",
     "https://www.crowd-counting.com/data/jhu_crowd_v2.0.zip",
     "binary"),
    ("dataset-file", "ShanghaiTech zip",
     "https://github.com/desenzhou/ShanghaiTechDataset/releases/download/v1.0/ShanghaiTech.zip",
     "binary"),

    # === Set B: doubao thread - landing/info pages ===
    ("landing", "S-HOCK DOI dataverse",
     "https://dataverse.osug.fr/dataset.xhtml?persistentId=doi:10.1007/978-3-319-16178-5_14",
     "page"),
    ("landing", "S-HOCK GitHub mirror",
     "https://github.com/MatteoTurchi/S-HOCK-Dataset",
     "page"),
    ("landing", "UCF-QNRF home",
     "https://www.crcv.ucf.edu/data/ucf-qnrf/",
     "page"),
    ("landing", "CrowdHuman home",
     "https://www.crowdhuman.org/",
     "page"),
    ("landing", "CrowdHuman download page",
     "https://www.crowdhuman.org/download.html",
     "page"),
    ("landing", "NWPU-Crowd home",
     "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/",
     "page"),
    ("landing", "JHU-CROWD home",
     "https://www.crowd-counting.com/",
     "page"),
    ("landing", "ShanghaiTech GitHub",
     "https://github.com/desenzhou/ShanghaiTechDataset",
     "page"),
    ("landing", "NWPU GitHub (gjy3035)",
     "https://github.com/gjy3035/NWPU-Crowd-Sample-Code",
     "page"),
    ("landing", "S-HOCK CVPR paper PDF",
     "https://openaccess.thecvf.com/content_cvpr_2015_workshops/w16/papers/Conigliaro_S-HOCK_A_Dataset_for_CVPR_2015_paper.pdf",
     "binary"),

    # === Set B: Hugging Face ===
    ("huggingface", "HF datasets index",
     "https://huggingface.co/datasets",
     "page"),
    ("huggingface", "HF crowdhuman",
     "https://huggingface.co/datasets/crowdhuman",
     "page"),
    ("huggingface", "HF stadium-crowd-detection",
     "https://huggingface.co/datasets/keremberke/stadium-crowd-detection",
     "page"),
    ("huggingface", "HF OpenGVLab crowd-counting",
     "https://huggingface.co/datasets/OpenGVLab/crowd-counting",
     "page"),
    ("huggingface", "HF mirror home",
     "https://hf-mirror.com/",
     "page"),
    ("huggingface", "HF mirror crowdhuman",
     "https://hf-mirror.com/datasets/crowdhuman",
     "page"),

    # === Set B: Kaggle ===
    ("kaggle", "Kaggle datasets index",
     "https://www.kaggle.com/datasets",
     "page"),
    ("kaggle", "Kaggle stadium-crowd-counting",
     "https://www.kaggle.com/datasets/robinreni/stadium-crowd-counting",
     "page"),
    ("kaggle", "Kaggle crowdhuman-dataset",
     "https://www.kaggle.com/datasets/odins0n/crowdhuman-dataset",
     "page"),

    # === Set B: Aliyun Tianchi ===
    ("tianchi", "Tianchi datasets index",
     "https://tianchi.aliyun.com/dataset",
     "page"),
    ("tianchi", "Tianchi NWPU-Crowd (107794)",
     "https://tianchi.aliyun.com/dataset/107794",
     "page"),
    ("tianchi", "Tianchi CrowdHuman (107793)",
     "https://tianchi.aliyun.com/dataset/107793",
     "page"),
    ("tianchi", "Tianchi UCF-QNRF (107795)",
     "https://tianchi.aliyun.com/dataset/107795",
     "page"),
    ("tianchi", "Tianchi ShanghaiTech (107796)",
     "https://tianchi.aliyun.com/dataset/107796",
     "page"),
    ("tianchi", "Tianchi JHU-CROWD (107797)",
     "https://tianchi.aliyun.com/dataset/107797",
     "page"),

    # === Set B: Baidu AI Studio ===
    ("aistudio", "AI Studio datasets index",
     "https://aistudio.baidu.com/aistudio/dataset",
     "page"),
    ("aistudio", "AI Studio NWPU (107794)",
     "https://aistudio.baidu.com/aistudio/datasetdetail/107794",
     "page"),
    ("aistudio", "AI Studio CrowdHuman (107793)",
     "https://aistudio.baidu.com/aistudio/datasetdetail/107793",
     "page"),
]


def probe(url, expected_type):
    """Send a Range GET and return a result dict."""
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": UA,
            "Range": "bytes=0-8191",
            "Accept": "*/*",
        },
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            elapsed_ms = int((time.time() - start) * 1000)
            status = resp.status
            final_url = resp.geturl()
            ctype = resp.headers.get("Content-Type", "").lower()
            clen = resp.headers.get("Content-Length", "?")
            body = resp.read(64)
            head_hex = body[:16].hex()
            is_html = "html" in ctype or body.lstrip().startswith(b"<")
            if expected_type == "binary":
                verdict = "OK" if not is_html else "WRONG"  # expected file, got HTML
            else:
                verdict = "OK"
            return {
                "verdict": verdict,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "ctype": ctype,
                "clen": clen,
                "head_hex": head_hex,
                "is_html": is_html,
                "final_url": final_url,
                "error": None,
            }
    except urllib.error.HTTPError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "verdict": "FAIL",
            "status": e.code,
            "elapsed_ms": elapsed_ms,
            "ctype": "",
            "clen": "?",
            "head_hex": "",
            "is_html": False,
            "final_url": url,
            "error": f"HTTP {e.code} {e.reason}",
        }
    except urllib.error.URLError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "verdict": "FAIL",
            "status": 0,
            "elapsed_ms": elapsed_ms,
            "ctype": "",
            "clen": "?",
            "head_hex": "",
            "is_html": False,
            "final_url": url,
            "error": f"URLError: {e.reason}",
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "verdict": "FAIL",
            "status": 0,
            "elapsed_ms": elapsed_ms,
            "ctype": "",
            "clen": "?",
            "head_hex": "",
            "is_html": False,
            "final_url": url,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    total = len(URLS)
    print(f"Probing {total} URLs with Range: bytes=0-8191 (timeout {TIMEOUT}s)\n")
    results = []

    for i, (cat, desc, url, etype) in enumerate(URLS, 1):
        host = urlparse(url).netloc
        print(f"[{i:>2}/{total}] {cat:<12} {host}")
        print(f"         {desc}")
        print(f"         {url}")
        r = probe(url, etype)
        results.append((cat, desc, url, etype, r))

        if r["verdict"] == "OK":
            label = "OK"
            if etype == "binary":
                label = f"OK (binary, {r['clen']} bytes)"
            else:
                label = f"OK (HTML page)"
            print(f"         -> [{label}] status={r['status']} {r['elapsed_ms']}ms ctype={r['ctype']}")
        elif r["verdict"] == "WRONG":
            print(f"         -> [LANDING-HTML instead of file] status={r['status']} {r['elapsed_ms']}ms ctype={r['ctype']}")
        else:
            print(f"         -> [FAIL] {r['error']} ({r['elapsed_ms']}ms)")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def sym(v, etype):
        if v == "OK":
            return "OK  "
        if v == "WRONG":
            return "WARN"
        return "FAIL"

    by_cat = {}
    for cat, desc, url, etype, r in results:
        by_cat.setdefault(cat, []).append((desc, url, etype, r))

    for cat, items in by_cat.items():
        print(f"\n--- {cat} ---")
        for desc, url, etype, r in items:
            s = sym(r["verdict"], etype)
            info = ""
            if r["verdict"] == "OK":
                info = f"{r['status']} {r['elapsed_ms']}ms"
            elif r["verdict"] == "WRONG":
                info = f"got HTML, wanted file ({r['status']})"
            else:
                info = r["error"] or "?"
            print(f"  [{s}] {desc:<40} {info}")
            print(f"         {url}")

    ok_count = sum(1 for _, _, _, _, r in results if r["verdict"] == "OK")
    warn_count = sum(1 for _, _, _, _, r in results if r["verdict"] == "WRONG")
    fail_count = sum(1 for _, _, _, _, r in results if r["verdict"] == "FAIL")
    print("\n" + "=" * 80)
    print(f"Total: {total}   OK: {ok_count}   WARN: {warn_count}   FAIL: {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
