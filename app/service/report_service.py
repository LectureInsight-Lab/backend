from idlelib.rpc import RemoteProxy

from app.core.store import get
from app.report.report_generator import generate
from app.schemas.report import ReportRequest

def generate_report(request: ReportRequest):
    scorecard = get(request.scorecard_id)

    if scorecard is None:
        raise ValueError("Scorecard를 찾을 수 없습니다.")

    generated_files = generate(scorecard=scorecard, formats=request.formats)

    return generated_files