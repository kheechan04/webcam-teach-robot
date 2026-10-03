"""data/lerobot/<이름> 데이터셋을 Hugging Face Hub에 비공개로 올린다. 라이선스는 붙이지 않는다(공개할 때 정한다).

    uv run python scripts/push_dataset.py cond2_webcam cond3_webcam_corrected cond4_depth_error cond4z_no_error
→ kheechan04/webcam-teach-robot-<이름 밑줄을 하이픈으로>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "kheechan04"


def repo_id(name: str) -> str:
    return f"{USER}/webcam-teach-robot-{name.replace('_', '-')}"


def main() -> None:
    from huggingface_hub import DatasetCard, HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    api = HfApi()
    for name in sys.argv[1:]:
        rid = repo_id(name)
        ds = LeRobotDataset(repo_id=rid, root=ROOT / "data" / "lerobot" / name)
        print(f"{name}: 에피소드 {ds.num_episodes}, 프레임 {ds.num_frames} → {rid}")
        ds.push_to_hub(private=True, license=None)
        try:  # 혹시 카드에 라이선스가 자동으로 붙었으면 지운다
            card = DatasetCard.load(rid, repo_type="dataset")
            if card.data.license:
                card.data.license = None
                card.push_to_hub(rid, repo_type="dataset", commit_message="Remove auto-added license (not decided yet)")
        except Exception as e:  # noqa: BLE001
            print("  카드 확인 실패:", e)
        info = api.dataset_info(rid)
        print(f"  올림: private={info.private}, license={getattr(info.card_data, 'license', None) if info.card_data else None}")


if __name__ == "__main__":
    main()
