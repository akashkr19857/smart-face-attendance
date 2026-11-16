"""Excel reports built from the database (never from the camera loop)."""
import io
import os

from loguru import logger
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_HEAD_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
_HEAD_FILL = PatternFill("solid", start_color="1F3A5F")
_BORDER = Border(*(Side(style="thin", color="D9D9D9"),) * 4)
_CENTER = Alignment(horizontal="center", vertical="center")
_OK_FILL = PatternFill("solid", start_color="C6EFCE")
_ALERT_FILL = PatternFill("solid", start_color="FFC7CE")


def _sheet(wb, title, headers, rows, widths, highlight_col=None, fill=None):
    ws = wb.create_sheet(title)
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font, cell.fill, cell.alignment, cell.border = _HEAD_FONT, _HEAD_FILL, _CENTER, _BORDER
        ws.column_dimensions[get_column_letter(c)].width = widths[c - 1]
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border, cell.alignment = _BORDER, _CENTER
        if highlight_col is not None:
            row[highlight_col].fill = fill
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 24


def build_workbook(storage, day, gate_id=None, camera_id=None):
    """Sheets: Attendance Records (counted per camera mode), Employee Summary, All Passages, Security Alerts."""
    wb = Workbook()
    wb.remove(wb.active)
    events = list(reversed(storage.events(day, gate_id, camera_id)))
    summary = storage.daily_summary(day, gate_id, camera_id)
    alerts = list(reversed(storage.alerts(day, gate_id, camera_id)))

    def event_row(i, e):
        return [i, day, e["ts"][11:], e["emp_id"], e["name"], e["department"], e["gate_name"], e["camera_id"],
                e["direction"], (e["attendance_mode"] or "").upper(), round(e["match_score"] * 100, 1),
                round(e["liveness"] * 100, 1)]

    event_head = ["S.No", "Date", "Time", "Employee ID", "Name", "Department", "Gate", "Camera", "Direction",
                  "Camera Mode", "Match %", "Liveness %"]
    event_w = [7, 12, 10, 14, 24, 18, 14, 18, 10, 13, 10, 11]
    _sheet(wb, "Attendance Records", event_head,
           [event_row(i, e) for i, e in enumerate([e for e in events if e["counted"]], 1)], event_w,
           highlight_col=9, fill=_OK_FILL)
    _sheet(wb, "Employee Summary",
           ["S.No", "Date", "Employee ID", "Name", "Department", "First Seen", "First Gate", "Last Seen",
            "Last Gate", "Passages", "Attendance Records", "Status"],
           [[i, day, s["emp_id"], s["name"], s["department"], s["first_seen"][11:], s["first_gate"],
             s["last_seen"][11:], s["last_gate"], s["passages"], s["records"], "PRESENT"]
            for i, s in enumerate(summary, 1)],
           [7, 12, 14, 24, 18, 12, 18, 12, 18, 10, 18, 12], highlight_col=11, fill=_OK_FILL)
    _sheet(wb, "All Passages", event_head + ["Counted"],
           [event_row(i, e) + ["YES" if e["counted"] else "repeat"] for i, e in enumerate(events, 1)], event_w + [10])
    _sheet(wb, "Security Alerts",
           ["S.No", "Time", "Type", "Gate", "Camera", "Match %", "Liveness %", "Detail", "Snapshot"],
           [[i, a["ts"][11:], a["kind"], a["gate_id"], a["camera_id"], round((a["match_score"] or 0) * 100, 1),
             round((a["liveness"] or 0) * 100, 1), a["detail"], a["snapshot"]] for i, a in enumerate(alerts, 1)],
           [7, 10, 12, 12, 18, 10, 11, 30, 50], highlight_col=2, fill=_ALERT_FILL)
    return wb


def workbook_bytes(storage, day, gate_id=None, camera_id=None):
    buf = io.BytesIO()
    build_workbook(storage, day, gate_id, camera_id).save(buf)
    return buf.getvalue()


def export_daily(storage, day, reports_dir):
    """Writes reports/attendance_<day>.xlsx atomically. Returns path, or None if the file is open in Excel."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"attendance_{day}.xlsx")
    tmp = path + ".tmp"
    build_workbook(storage, day).save(tmp)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        os.remove(tmp)
        logger.warning("{} is open in another program - will retry on next autosave", path)
        return None
