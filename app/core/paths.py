"""외부 데이터 경로 로더.

`configs/paths.yaml` 의 경로를 Pydantic 모델로 노출.
STT 원본/메타데이터는 repo 외부에 두고 경로 참조 방식으로 접근한다.
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"


class DataPaths(BaseModel):
    stt_dir: Path
    metadata_csv: Path


class PathsConfig(BaseModel):
    data: DataPaths
    stt_filename_pattern: str = "{date}_{course_id}.txt"

    def stt_file(self, lecture_date: str, course_id: str) -> Path:
        """STT 파일 절대 경로."""
        filename = self.stt_filename_pattern.format(date=lecture_date, course_id=course_id)
        return self.data.stt_dir / filename


@lru_cache(maxsize=1)
def load_paths(config_path: Path = DEFAULT_PATHS_CONFIG) -> PathsConfig:
    """`configs/paths.yaml` 을 로드해 PathsConfig 반환 (캐싱)."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"paths config not found: {config_path}\n"
            f"configs/paths.yaml 을 작성하거나 paths.example.yaml 을 복사하세요."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PathsConfig.model_validate(raw)


def read_stt(lecture_date: str, course_id: str) -> str:
    """주어진 날짜+코스의 STT 원본 텍스트를 읽어 반환."""
    cfg = load_paths()
    path = cfg.stt_file(lecture_date, course_id)
    if not path.exists():
        raise FileNotFoundError(f"STT 파일을 찾지 못함: {path}")
    return path.read_text(encoding="utf-8")
