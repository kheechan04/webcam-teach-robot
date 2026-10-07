"""Hub 공개 (2026-10-07 사용자 결정): 데이터셋 14개 CC BY 4.0, 최종 비교 모델 35개 Apache-2.0, 나머지 모델은 비공개 유지.

    uv run python scripts/publish_hub.py --dry     # 무엇을 바꿀지 보기만
    uv run python scripts/publish_hub.py           # 카드 라이선스를 달고 공개로 바꾼다
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, metadata_update

USER = "kheechan04"
DATASETS = ["cond1-scripted", "cond2-webcam", "cond3-webcam-corrected", "cond4-depth-error", "cond4z-no-error",
            "cond4h-error-half", "cond4d-error-double", "cond5-webcam-marker", "cond2-p2", "cond3-p2",
            "stack1-scripted", "stack2-webcam", "stack3-webcam-corrected", "m9-results"]
MODELS = ([f"act-{c}-s{s}-100k" for c in ("cond1-scripted", "cond4-depth-error", "cond4z-no-error", "cond5-webcam-marker")
           for s in (1000, 2000, 3000)]
          + [f"act-{c}-s{s}-100k" for c in ("cond2-webcam", "cond3-webcam-corrected") for s in (1000, 2000, 3000, 4000, 5000)]
          + [f"act-stack1-scripted-s{s}-20k" for s in (1000, 2000, 3000)]
          + [f"act-{c}-s{s}-60k" for c in ("stack2-webcam", "stack3-webcam-corrected") for s in (1000, 2000, 3000, 4000, 5000)])
NOTE = ("\n\nPart of [webcam-teach-robot](https://github.com/kheechan04/webcam-teach-robot): teaching a simulated SO-101 arm "
        "from webcam hand demonstrations and measuring how monocular depth error affects imitation learning. "
        "No webcam video or images of people: numbers and simulation renders only.\n")


def main() -> None:
    dry = "--dry" in sys.argv
    api = HfApi()
    assert len(DATASETS) == 14 and len(MODELS) == 35
    for name in DATASETS:
        rid = f"{USER}/webcam-teach-robot-{name}"
        files = api.list_repo_files(rid, repo_type="dataset")
        print(("[dry] " if dry else "") + f"dataset {rid}: README {'있음' if 'README.md' in files else '없음'} → cc-by-4.0, 공개")
        if dry:
            continue
        if "README.md" not in files:  # 평가 결과 저장소에는 카드가 없었다
            api.upload_file(path_or_fileobj=("---\nlicense: cc-by-4.0\n---\n# webcam-teach-robot evaluation results\n\n"
                                             "Evaluation JSONs (`eval/`) and training-server logs (`logs/`)." + NOTE).encode(),
                            path_in_repo="README.md", repo_id=rid, repo_type="dataset", commit_message="Add card (CC BY 4.0)")
        else:
            metadata_update(rid, {"license": "cc-by-4.0"}, repo_type="dataset", overwrite=True)
            body = Path(hf_hub_download(rid, "README.md", repo_type="dataset", force_download=True)).read_text(encoding="utf-8")
            new = (body.replace("- **License:** [More Information Needed]", "- **License:** CC BY 4.0")
                   .replace("- **Homepage:** [More Information Needed]", "- **Homepage:** https://github.com/kheechan04/webcam-teach-robot"))
            if NOTE.strip() not in new:
                new = new + NOTE
            if new != body:
                api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md", repo_id=rid, repo_type="dataset",
                                commit_message="Card: license and homepage")
        api.update_repo_settings(rid, repo_type="dataset", private=False)
    for name in MODELS:
        rid = f"{USER}/webcam-teach-robot-{name}"
        api.model_info(rid)
        print(("[dry] " if dry else "") + f"model {rid} → apache-2.0, 공개")
        if dry:
            continue
        metadata_update(rid, {"license": "apache-2.0"}, overwrite=True)
        body = Path(hf_hub_download(rid, "README.md", force_download=True)).read_text(encoding="utf-8")
        if NOTE.strip() not in body:
            api.upload_file(path_or_fileobj=(body + NOTE).encode(), path_in_repo="README.md", repo_id=rid,
                            commit_message="Card: project note")
        api.update_repo_settings(rid, private=False)


if __name__ == "__main__":
    main()
