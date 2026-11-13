"""K리그 선수 기록을 분석하는 Streamlit 기반 AI 스카우팅 앱."""

# =========================================================
# K리그/유소년 AI 스카우팅 통합본
# [업데이트 완료]: 연도별 + 포지션별 완벽 독립 Z-score 연산 엔진 적용
# [추가 업데이트]: 선택 선수 연도별 AI 스카우팅 스코어 트렌드 분석 창 적용
# =========================================================

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from analysis_utils import standardize_metrics, recommend_players, get_player_history
from sample_data import make_sample_data

st.set_page_config(page_title="AI 스카우팅 플랫폼", layout="wide", page_icon="⚽")
st.title("⚽ 머신러닝 기반 AI 스카우팅 플랫폼")
st.info("점수·등급은 사용자 가중치에 따른 시즌·포지션 그룹 내 상대평가입니다. 검증된 절대 능력이나 미래 성과 예측이 아니며 0~100으로 제한되지 않습니다.")
st.caption("포지션·플레이스타일은 규칙 기반 추정입니다.")

# =========================================================
# 1. 지표 매핑 데이터베이스
# =========================================================
feature_korean = {
    "Goal_Qty": "총 득점", "Per_Goal_Qty": "90분당 득점",
    "St_Qty": "총 슈팅", "Per_St_Qty": "90분당 슈팅",
    "St_Valid_Qty": "유효슈팅", "Per_St_Valid_Qty": "90분당 유효슈팅",
    "Assistant_Qty": "도움", "Assist_Qty": "도움", "Per_Assist_Qty": "90분당 도움",
    "fs_qty": "피파울", "Per_fs_qty": "90분당 피파울",
    "SHOT": "슈팅 시도", "Per_SHOT": "90분당 슈팅 시도",
    "SHOT_ON_TARGET": "유효슈팅", "Per_SHOT_ON_TARGET": "90분당 유효슈팅",
    "DEFENDER_BLOCKED_SHOT": "수비수에게 막힌 슈팅", "Per_DEFENDER_BLOCKED_SHOT": "90분당 막힌 슈팅",
    "SHOOTING_OFF": "골대 밖 슈팅", "Per_SHOOTING_OFF": "90분당 골대 밖 슈팅",
    "SHOOTING_IN_PA": "PA 안 슈팅", "Per_SHOOTING_IN_PA": "90분당 PA 안 슈팅",
    "SHOOTING_OUT_PA": "PA 밖 슈팅", "Per_SHOOTING_OUT_PA": "90분당 PA 밖 슈팅",
    "SHOT_ON_TARGET_IN_PA": "PA 안 유효슈팅", "Per_SHOT_ON_TARGET_IN_PA": "90분당 PA 안 유효슈팅",
    "OFFSIDE": "오프사이드", "Per_OFFSIDE": "90분당 오프사이드",
    "DRIBBLE": "드리블 시도", "Per_DRIBBLE": "90분당 드리블 시도",
    "DRIBBLE_SUCCESS": "드리블 성공", "Per_DRIBBLE_SUCCESS": "90분당 드리블 성공",
    "PCNT_SUCCESS_DRIBBLE": "드리블 성공률",
    "CONTROL_UNDER_PRESSURE": "압박 상황 볼 컨트롤", "Per_CONTROL_UNDER_PRESSURE": "90분당 압박 상황 컨트롤",
    "BALL_MISS": "볼 미스", "Per_BALL_MISS": "90분당 볼 미스",
    "FOUL": "파울", "Per_FOUL": "90분당 파울",
    "warn_qty": "경고", "Per_warn_qty": "90분당 경고",
    "exit_qty": "퇴장", "Per_exit_qty": "90분당 퇴장",
    "PASS": "패스 시도", "Per_PASS": "90분당 패스 시도",
    "PASS_ACC": "패스 성공", "Per_PASS_ACC": "90분당 패스 성공",
    "PCNT_ACC_PASS": "패스 성공률",
    "EX_ATT_KEY_PASS": "키패스 시도", "Per_EX_ATT_KEY_PASS": "90분당 키패스 시도",
    "EX_ATT_KEY_PASS_ACC": "키패스 성공", "Per_EX_ATT_KEY_PASS_ACC": "90분당 키패스 성공",
    "PASS_F_CA_DEG": "전방패스 시도", "Per_PASS_F_CA_DEG": "90분당 전방패스 시도",
    "PASS_F_ACC_CA_DEG": "전방패스 성공", "Per_PASS_F_ACC_CA_DEG": "90분당 전방패스 성공",
    "PCNT_PASS_F_CA_DEG": "전방패스 성공률",
    "PASS_B_CA_DEG": "후방패스 시도", "Per_PASS_B_CA_DEG": "90분당 후방패스 시도",
    "PASS_B_ACC_CA_DEG": "후방패스 성공", "Per_PASS_B_ACC_CA_DEG": "90분당 후방패스 성공",
    "PCNT_PASS_B_ACC_CA_DEG": "후방패스 성공률",
    "TRANSVERSE_PASS_SUCCESS": "횡패스 성공", "Per_TRANSVERSE_PASS_SUCCESS": "90분당 횡패스 성공",
    "ATT_AREA_PASS": "공격지역 패스 시도", "Per_ATT_AREA_PASS": "90분당 공격지역 패스 시도",
    "ATT_AREA_PASS_SUCCESS": "공격지역 패스 성공", "Per_ATT_AREA_PASS_SUCCESS": "90분당 공격지역 패스 성공",
    "PCNT_ATT_AREA_PASS_SUCCESS": "공격지역 패스 성공률",
    "MIDDLE_AREA_PASS_SUCCESS": "중원지역 패스 성공", "Per_MIDDLE_AREA_PASS_SUCCESS": "90분당 중원지역 패스 성공",
    "LONG_PASS": "롱패스 시도", "Per_LONG_PASS": "90분당 롱패스 시도",
    "LONG_PASS_SUCCESS": "롱패스 성공", "Per_LONG_PASS_SUCCESS": "90분당 롱패스 성공",
    "PCNT_LONG_PASS_SUCCESS": "롱패스 성공률",
    "MEDIUM_PASS_SUCCESS": "중거리 패스 성공",
    "SHORT_PASS_SUCCESS": "짧은 패스 성공", "Per_SHORT_PASS_SUCCESS": "90분당 짧은 패스 성공",
    "CHALG": "지상경합 시도", "Per_CHALG": "90분당 지상경합 시도",
    "CHALG_WON": "지상경합 승리", "Per_CHALG_WON": "90분당 지상경합 승리",
    "PCNT_CHALG_WON": "지상경합 승률",
    "AIR_CHALG": "공중볼 경합 시도", "Per_AIR_CHALG": "90분당 공중볼 경합 시도",
    "AIR_CHALG_WON": "공중볼 경합 승리", "Per_AIR_CHALG_WON": "90분당 공중볼 경합 승리",
    "AIR_CHALLENGE_PCNT": "공중볼 경합 승률",
    "TACKLE": "태클 시도", "Per_TACKLE": "90분당 태클 시도",
    "TACKLE_SUCCESS": "태클 성공", "Per_TACKLE_SUCCESS": "90분당 태클 성공",
    "PCNT_TACKLE_WON": "태클 성공률",
    "INTERCEPT": "인터셉트", "Per_INTERCEPT": "90분당 인터셉트",
    "CUT_OFF": "컷오프", "Per_CUT_OFF": "90분당 컷오프",
    "BALL_RECOVERIE": "볼 리커버리", "Per_BALL_RECOVERIE": "90분당 볼 리커버리",
    "KICK_OUT": "킥아웃", "Per_KICK_OUT": "90분당 킥아웃",
    "DEFS_AREA_PASS": "수비지역 패스 시도", "Per_DEFS_AREA_PASS": "90분당 수비지역 패스 시도",
    "DEFS_AREA_PASS_SUCCESS": "수비지역 패스 성공", "Per_DEFS_AREA_PASS_SUCCESS": "90분당 수비지역 패스 성공",
    "PCNT_DEFS_AREA_PASS_SUCCESS": "수비지역 패스 성공률",
    "BLOCKING_BALL": "블락", "Per_BLOCKING_BALL": "90분당 블락"
}

negative_features = [
    "DEFENDER_BLOCKED_SHOT", "Per_DEFENDER_BLOCKED_SHOT", "SHOOTING_OFF", "Per_SHOOTING_OFF",
    "OFFSIDE", "Per_OFFSIDE", "BALL_MISS", "Per_BALL_MISS", "FOUL", "Per_FOUL",
    "warn_qty", "Per_warn_qty", "exit_qty", "Per_exit_qty"
]

per90_map = {
    "Goal_Qty": "Per_Goal_Qty", "St_Qty": "Per_St_Qty", "St_Valid_Qty": "Per_St_Valid_Qty", "Assist_Qty": "Per_Assist_Qty",
    "fs_qty": "Per_fs_qty", "SHOT": "Per_SHOT", "SHOT_ON_TARGET": "Per_SHOT_ON_TARGET", "DEFENDER_BLOCKED_SHOT": "Per_DEFENDER_BLOCKED_SHOT",
    "SHOOTING_OFF": "Per_SHOOTING_OFF", "SHOOTING_IN_PA": "Per_SHOOTING_IN_PA", "SHOOTING_OUT_PA": "Per_SHOOTING_OUT_PA",
    "SHOT_ON_TARGET_IN_PA": "Per_SHOT_ON_TARGET_IN_PA", "OFFSIDE": "Per_OFFSIDE", "DRIBBLE": "Per_DRIBBLE",
    "DRIBBLE_SUCCESS": "Per_DRIBBLE_SUCCESS", "CONTROL_UNDER_PRESSURE": "Per_CONTROL_UNDER_PRESSURE",
    "BALL_MISS": "Per_BALL_MISS", "FOUL": "Per_FOUL", "PASS": "Per_PASS", "PASS_ACC": "Per_PASS_ACC",
    "EX_ATT_KEY_PASS": "Per_EX_ATT_KEY_PASS", "EX_ATT_KEY_PASS_ACC": "Per_EX_ATT_KEY_PASS_ACC",
    "PASS_F_CA_DEG": "Per_PASS_F_CA_DEG", "PASS_F_ACC_CA_DEG": "Per_PASS_F_ACC_CA_DEG", "PASS_B_CA_DEG": "Per_PASS_B_CA_DEG",
    "PASS_B_ACC_CA_DEG": "Per_PASS_B_ACC_CA_DEG", "TRANSVERSE_PASS_SUCCESS": "Per_TRANSVERSE_PASS_SUCCESS",
    "ATT_AREA_PASS": "Per_ATT_AREA_PASS", "ATT_AREA_PASS_SUCCESS": "Per_ATT_AREA_PASS_SUCCESS",
    "MIDDLE_AREA_PASS_SUCCESS": "Per_MIDDLE_AREA_PASS_SUCCESS", "LONG_PASS": "Per_LONG_PASS",
    "LONG_PASS_SUCCESS": "Per_LONG_PASS_SUCCESS", "MEDIUM_PASS_SUCCESS": "Per_MEDIUM_PASS_SUCCESS",
    "SHORT_PASS_SUCCESS": "Per_SHORT_PASS_SUCCESS", "CHALG": "Per_CHALG", "CHALG_WON": "Per_CHALG_WON",
    "AIR_CHALG": "Per_AIR_CHALG", "AIR_CHALG_WON": "Per_AIR_CHALG_WON", "TACKLE": "Per_TACKLE",
    "TACKLE_SUCCESS": "Per_TACKLE_SUCCESS", "PCNT_TACKLE_WON": "Teckle_Won_Rate", "INTERCEPT": "Per_INTERCEPT",
    "CUT_OFF": "Per_CUT_OFF", "BALL_RECOVERIE": "Per_BALL_RECOVERIE", "KICK_OUT": "Per_KICK_OUT",
    "DEFS_AREA_PASS": "Per_DEFS_AREA_PASS", "DEFS_AREA_PASS_SUCCESS": "Per_DEFS_AREA_PASS_SUCCESS", "BLOCKING_BALL": "Per_BLOCKING_BALL",
    "warn_qty": "Per_warn_qty", "exit_qty": "Per_exit_qty"
}

defensive_metrics = {
    "CHALG", "CHALG_WON", "PCNT_CHALG_WON", "AIR_CHALG", "AIR_CHALG_WON", "AIR_CHALLENGE_PCNT",
    "TACKLE", "TACKLE_SUCCESS", "PCNT_TACKLE_WON", "INTERCEPT", "CUT_OFF", "BALL_RECOVERIE",
    "KICK_OUT", "BLOCKING_BALL", "DEFS_AREA_PASS", "DEFS_AREA_PASS_SUCCESS", "PCNT_DEFS_AREA_PASS_SUCCESS"
}
buildup_metrics = {
    "PASS", "PASS_ACC", "PCNT_ACC_PASS", "PASS_F_CA_DEG", "PASS_F_ACC_CA_DEG", "PCNT_PASS_F_CA_DEG",
    "PASS_B_CA_DEG", "PASS_B_ACC_CA_DEG", "PCNT_PASS_B_ACC_CA_DEG", "TRANSVERSE_PASS_SUCCESS",
    "MIDDLE_AREA_PASS_SUCCESS", "LONG_PASS", "LONG_PASS_SUCCESS", "PCNT_LONG_PASS_SUCCESS",
    "MEDIUM_PASS_SUCCESS", "SHORT_PASS_SUCCESS"
}

# =========================================================
# 2. 기본 가중치 구성 프로필
# =========================================================
base_attack_weights = {
    "Goal_Qty": 0.12, "St_Qty": 0.04, "St_Valid_Qty": 0.07, "Assist_Qty": 0.05,
    "fs_qty": 0.03, "SHOT": 0.03, "SHOT_ON_TARGET": 0.07, "DEFENDER_BLOCKED_SHOT": 0.03,
    "SHOOTING_OFF": 0.03, "SHOOTING_IN_PA": 0.06, "SHOOTING_OUT_PA": 0.02,
    "SHOT_ON_TARGET_IN_PA": 0.06, "OFFSIDE": 0.02, "DRIBBLE": 0.03,
    "DRIBBLE_SUCCESS": 0.05, "PCNT_SUCCESS_DRIBBLE": 0.04, "CONTROL_UNDER_PRESSURE": 0.04,
    "BALL_MISS": 0.05, "FOUL": 0.03, "warn_qty": 0.03, "exit_qty": 0.08
}
base_midfielder_weights = {
    "PASS": 0.03, "PASS_ACC": 0.05, "PCNT_ACC_PASS": 0.06, "EX_ATT_KEY_PASS": 0.06,
    "EX_ATT_KEY_PASS_ACC": 0.07, "PASS_F_CA_DEG": 0.04, "PASS_F_ACC_CA_DEG": 0.06,
    "PCNT_PASS_F_CA_DEG": 0.05, "PASS_B_CA_DEG": 0.01, "PASS_B_ACC_CA_DEG": 0.02,
    "PCNT_PASS_B_ACC_CA_DEG": 0.02, "TRANSVERSE_PASS_SUCCESS": 0.03, "ATT_AREA_PASS": 0.04,
    "ATT_AREA_PASS_SUCCESS": 0.05, "PCNT_ATT_AREA_PASS_SUCCESS": 0.04, "MIDDLE_AREA_PASS_SUCCESS": 0.04,
    "LONG_PASS": 0.03, "LONG_PASS_SUCCESS": 0.04, "PCNT_LONG_PASS_SUCCESS": 0.04,
    "MEDIUM_PASS_SUCCESS": 0.03, "SHORT_PASS_SUCCESS": 0.03, "DRIBBLE": 0.03,
    "DRIBBLE_SUCCESS": 0.04, "PCNT_SUCCESS_DRIBBLE": 0.04, "CONTROL_UNDER_PRESSURE": 0.04,
    "BALL_MISS": 0.05, "CHALG": 0.02, "CHALG_WON": 0.03, "PCNT_CHALG_WON": 0.03,
    "AIR_CHALG": 0.02, "AIR_CHALG_WON": 0.02, "AIR_CHALLENGE_PCNT": 0.02, "TACKLE": 0.02,
    "TACKLE_SUCCESS": 0.03, "PCNT_TACKLE_WON": 0.03, "INTERCEPT": 0.04, "CUT_OFF": 0.03,
    "BALL_RECOVERIE": 0.04, "FOUL": 0.02, "warn_qty": 0.02, "exit_qty": 0.06
}
base_defender_weights = {
    "CHALG": 0.04, "CHALG_WON": 0.07, "PCNT_CHALG_WON": 0.06, "AIR_CHALG": 0.05,
    "AIR_CHALG_WON": 0.08, "AIR_CHALLENGE_PCNT": 0.07, "TACKLE": 0.04, "TACKLE_SUCCESS": 0.07,
    "PCNT_TACKLE_WON": 0.06, "INTERCEPT": 0.07, "CUT_OFF": 0.05, "BALL_RECOVERIE": 0.05,
    "KICK_OUT": 0.04, "DEFS_AREA_PASS": 0.03, "DEFS_AREA_PASS_SUCCESS": 0.04,
    "PCNT_DEFS_AREA_PASS_SUCCESS": 0.04, "BLOCKING_BALL": 0.06, "BALL_MISS": 0.06,
    "FOUL": 0.02, "warn_qty": 0.02, "exit_qty": 0.06
}

# =========================================================
# 3. 데이터 로딩 및 공통 파이프라인 함수
# =========================================================
@st.cache_data
def load_data(file):
    """Load a CSV or Excel player-statistics file and standardize its columns."""
    file.seek(0)
    file_name = file.name.lower()
    column_translation = {}

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")
    elif file_name.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if file_name.endswith(".xlsx") else "xlrd"
        file.seek(0)
        # 첫 번째 시트를 선수 기록 데이터로 읽습니다.
        excel = pd.ExcelFile(file, engine=engine)
        df = pd.read_excel(excel, sheet_name=0)

        # 한글 컬럼명 파일이면 '컬럼명_변환표' 시트를 이용해
        # 앱 내부에서 사용하는 원래 영문 지표명으로 복원합니다.
        if "컬럼명_변환표" in excel.sheet_names:
            mapping_df = pd.read_excel(excel, sheet_name="컬럼명_변환표")
            required_mapping_cols = {"기존 컬럼명", "변경된 한글 컬럼명"}
            if required_mapping_cols.issubset(mapping_df.columns):
                korean_to_original = dict(zip(
                    mapping_df["변경된 한글 컬럼명"].astype(str).str.strip(),
                    mapping_df["기존 컬럼명"].astype(str).str.strip()
                ))
                # 앱에서 한글로 사용하는 식별자 열은 한글 상태로 유지합니다.
                identifier_cols = {"연도", "팀명", "선수명", "포지션", "출전시간"}
                column_translation = {
                    col: korean_to_original[col]
                    for col in df.columns
                    if col in korean_to_original
                    and col not in identifier_cols
                }
                df = df.rename(columns=column_translation)
    else:
        raise ValueError("CSV, XLSX, XLS 파일만 업로드할 수 있습니다.")

    # 영문 원본 파일의 식별자 열도 앱 표준 한글 이름으로 맞춥니다.
    df = df.rename(columns={
        "year": "연도",
        "team_name": "팀명",
        "player_name": "선수명",
        "POSITION_NM": "포지션",
        "work_time": "출전시간"
    })

    # 서로 다른 원본 열을 같은 이름으로 합쳐 중복 열이 생기는 것을 방지합니다.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 일부 파일에 있는 WORK_TIME은 work_time과 형식이 다른 설명용 열이므로
    # 숫자형 출전시간으로 오인하지 않도록 제거합니다.
    if "WORK_TIME" in df.columns:
        df = df.drop(columns=["WORK_TIME"])

    text_cols = ["연도", "팀명", "선수명", "포지션"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", "None", "NaN", "NAT", "<NA>"], "")

    if "연도" in df.columns:
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").fillna(0).astype(int).astype(str)
        df["연도"] = df["연도"].replace("0", "")

    # 필수 식별자 열이 없으면 원인을 알기 쉽게 안내합니다.
    missing_ids = [col for col in ["선수명", "팀명"] if col not in df.columns]
    if missing_ids:
        raise ValueError(
            f"필수 열이 없습니다: {', '.join(missing_ids)}. "
            "업로드 파일에 선수명과 팀명 열이 있는지 확인해 주세요."
        )

    df = df[(df["선수명"] != "") & (df["팀명"] != "")]
    for col in df.columns:
        if col not in text_cols and col != "출전시간":
            df[col] = df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "출전시간" in df: df["출전시간"] = pd.to_numeric(df["출전시간"], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def classify_position_group(pos):
    """Map a player's listed position to a broad position group."""
    pos = str(pos).upper()
    if any(x in pos for x in ["FW", "ST", "CF", "WF", "L/RW", "RW", "LW"]): return "ATTACK"
    elif any(x in pos for x in ["MF", "AM", "CM", "DM", "CAM", "CDM"]): return "MIDFIELDER"
    elif any(x in pos for x in ["DF", "CB", "LB", "RB", "WB", "L/RB"]): return "DEFENDER"
    return "ETC"

def _get_z(row, base_col, data_mode):
    """Return the Z-score column value for the selected data mode."""
    if data_mode == "90분당 데이터" and base_col in per90_map:
        target_col = per90_map[base_col] + "_Z"
    else:
        target_col = base_col + "_Z"
    return float(row.get(target_col, 0.0))

def determine_detailed_position(row, data_mode):
    """Infer a detailed playing position from standardized metrics."""
    g = row["POSITION_GROUP"]
    z = lambda c: _get_z(row, c, data_mode)
    if g == "ATTACK":
        st_score = (z("SHOT_ON_TARGET_IN_PA") * 2.0) + (z("Goal_Qty") * 1.5) + (z("AIR_CHALG_WON") * 1.0)
        lrw_score = (z("DRIBBLE_SUCCESS") * 2.0) + (z("ATT_AREA_PASS_SUCCESS") * 1.5) + (z("PASS_F_ACC_CA_DEG") * 1.0)
        return "ST" if st_score >= lrw_score else "L/RW"
    elif g == "MIDFIELDER":
        cam = (z("EX_ATT_KEY_PASS_ACC") * 2.0) + (z("Assist_Qty") * 1.5) + (z("SHOOTING_IN_PA") * 1.0)
        cdm = (z("INTERCEPT") * 1.8) + (z("TACKLE_SUCCESS") * 1.5) + (z("CUT_OFF") * 1.2) + (z("DEFS_AREA_PASS_SUCCESS") * 0.8)
        cm = (z("PCNT_ACC_PASS") * 1.5) + (z("MIDDLE_AREA_PASS_SUCCESS") * 1.5) + (z("TRANSVERSE_PASS_SUCCESS") * 1.0)
        if cdm > cam and cdm > cm: return "CDM"
        return "CAM" if cam >= cm else "CM"
    elif g == "DEFENDER":
        cb = (z("BLOCKING_BALL") * 2.0) + (z("KICK_OUT") * 1.5) + (z("AIR_CHALLENGE_PCNT") * 1.2)
        lrb = (z("DRIBBLE") * 1.5) + (z("ATT_AREA_PASS") * 1.5) + (z("PASS_F_CA_DEG") * 1.2)
        return "CB" if cb >= lrb else "L/RB"
    return str(row.get("포지션", "ETC"))

def assign_composite_roles(row, data_mode):
    """Calculate likely playing styles from standardized metrics."""
    g = row["POSITION_GROUP"]
    z = lambda c: _get_z(row, c, data_mode)
    roles = {}
    if g == "ATTACK":
        roles["포쳐(스코어러)"] = z("Goal_Qty") * 1.5 + z("St_Valid_Qty") * 1.0
        roles["타겟맨"] = z("AIR_CHALG_WON") * 1.5 + z("SHOT_ON_TARGET_IN_PA") * 1.0
        roles["연계형 포워드"] = z("Assist_Qty") * 1.2 + z("EX_ATT_KEY_PASS") * 1.0 + z("PASS_F_ACC_CA_DEG") * 0.8
        roles["인사이드 포워드"] = z("DRIBBLE_SUCCESS") * 1.2 + z("SHOOTING_IN_PA") * 1.0 + z("SHOT") * 0.8
        roles["클래식 윙어"] = z("ATT_AREA_PASS") * 1.5 + z("DRIBBLE") * 1.0
    elif g == "MIDFIELDER":
        roles["전진형 플레이메이커"] = z("EX_ATT_KEY_PASS_ACC") * 1.5 + z("ATT_AREA_PASS_SUCCESS") * 1.0 + z("Assist_Qty") * 1.0
        roles["후방 플레이메이커"] = z("LONG_PASS_SUCCESS") * 1.2 + z("PASS_F_ACC_CA_DEG") * 1.2 + z("PCNT_ACC_PASS") * 0.6
        roles["박스 투 박스"] = z("CHALG_WON") * 1.0 + z("BALL_RECOVERIE") * 1.0 + z("MIDDLE_AREA_PASS_SUCCESS") * 1.0
        roles["홀딩 파이터"] = z("TACKLE_SUCCESS") * 1.2 + z("INTERCEPT") * 1.0 + z("CUT_OFF") * 0.8
    elif g == "DEFENDER":
        roles["빌드업 센터백"] = z("PASS_F_ACC_CA_DEG") * 1.5 + z("DEFS_AREA_PASS_SUCCESS") * 1.0
        roles["파이터형 스토퍼"] = z("TACKLE_SUCCESS") * 1.2 + z("CHALG_WON") * 1.0 + z("BLOCKING_BALL") * 0.8
        roles["제공권 장악형"] = z("AIR_CHALG_WON") * 1.5 + z("KICK_OUT") * 1.0
        roles["공격형 풀백"] = z("ATT_AREA_PASS") * 1.2 + z("DRIBBLE") * 1.0 + z("PASS_F_CA_DEG") * 0.8
        roles["수비형 풀백"] = z("INTERCEPT") * 1.5 + z("CUT_OFF") * 1.5 - (z("ATT_AREA_PASS") * 0.5)
    else:
        return "분석 불가", "분석 불가"

    T = 1.5
    max_s = max(roles.values())
    exp_s = {k: math.exp((v - max_s) / T) for k, v in roles.items()}
    total = sum(exp_s.values())
    probs = {k: (v / total) * 100 for k, v in exp_s.items()}
    srt = sorted(probs.items(), key=lambda i: i[1], reverse=True)
    primary = srt[0][0]
    comp = f"[{srt[0][0]} {srt[0][1]:.1f}%] + [{srt[1][0]} {srt[1][1]:.1f}%] + [{srt[2][0]} {srt[2][1]:.1f}%]"
    return primary, comp

def allowed_weakness(col, group):
    """Check whether a weakness metric is relevant to the position group."""
    base = col.replace("Per_", "")
    if group == "DEFENDER":
        return base in defensive_metrics or base in buildup_metrics
    if group == "ATTACK":
        return base not in defensive_metrics
    return True

def analyze_strength_weakness(sub, weights, group):
    """Summarize unusually strong and weak metrics for each player."""
    means = {c: sub[c].mean() for c in weights}
    def sw(row):
        strong, weak = [], []
        for c in weights:
            m = means[c]
            if m == 0 or pd.isna(m):
                continue
            val = float(row[c])
            ratio = val / m if m != 0 else 0
            is_neg = c in negative_features or c.replace("Per_", "") in negative_features
            name = feature_korean.get(c, c)
            if is_neg:
                if ratio <= 0.7: strong.append((name, ratio))
                elif ratio >= 1.5: weak.append((name, ratio, c))
            else:
                if ratio >= 1.5: strong.append((name, ratio))
                elif ratio <= 0.7: weak.append((name, ratio, c))
        weak = [w for w in weak if allowed_weakness(w[2], group)]
        strong = sorted(strong, key=lambda x: -x[1])[:5]
        weak = sorted(weak, key=lambda x: x[1])[:5]
        s_str = ", ".join([n for n, _ in strong]) if strong else "특이 강점 없음"
        w_str = ", ".join([n for n, _, _ in weak]) if weak else "특이 약점 없음"
        return pd.Series([s_str, w_str])
    sub[["강점", "약점"]] = sub.apply(sw, axis=1)
    return sub

def get_grade(score):
    """Convert a scouting score into a letter grade."""
    scr = round(float(score), 2)
    if scr >= 85.0: return "S+"
    elif scr >= 75.0: return "S"
    elif scr >= 68.0: return "A+"
    elif scr >= 62.0: return "A"
    elif scr >= 55.0: return "B+"
    elif scr >= 48.0: return "B"
    elif scr >= 42.0: return "C+"
    elif scr >= 36.0: return "C"
    elif scr >= 30.0: return "D+"
    return "D"

def convert_weights_by_mode(weights, data_mode, data):
    """Map configured weights to cumulative or per-90 columns."""
    conv = {}
    for raw, w in weights.items():
        if data_mode == "90분당 데이터":
            per = per90_map.get(raw)
            if per in data.columns: conv[per] = w
            elif raw in data.columns: conv[raw] = w
        else:
            if raw in data.columns: conv[raw] = w
    return conv

def normalize_weights(weights):
    """Normalize weights so their absolute values sum to one."""
    total = sum(abs(v) for v in weights.values())
    if total == 0: return weights
    return {k: v / total for k, v in weights.items()}

def add_zscores(data, weights):
    """Add standardized metric columns to a copy of the data."""
    data = data.copy()
    for col in weights:
        mu, sd = data[col].mean(), data[col].std()
        data[col + "_Z"] = 0 if (sd == 0 or pd.isna(sd)) else (data[col] - mu) / sd
    return data

def calculate_score(data, weights):
    """Calculate scouting scores and grades from weighted Z-scores."""
    data = data.copy()
    data["RAW_WEIGHTED_SCORE"] = 0
    for col, w in weights.items():
        z = col + "_Z"
        if col in negative_features or col.replace("Per_", "") in negative_features:
            data["RAW_WEIGHTED_SCORE"] -= data[z] * abs(w)
        else:
            data["RAW_WEIGHTED_SCORE"] += data[z] * abs(w)
    data["SCOUTING_SCORE"] = (data["RAW_WEIGHTED_SCORE"] * 20 + 50).round(2)
    data["GRADE"] = data["SCOUTING_SCORE"].apply(get_grade)
    return data

# 🌟 [연도별 독립 연산 최적화]: 연도별 루프 안에서 다시 포지션 그룹별로 쪼개 계산합니다.
def apply_model(data, data_mode, w_att, w_mid, w_def):
    """Run the scouting model independently by season and position group."""
    if len(data) == 0: return pd.DataFrame()

    distinct_years = data["연도"].unique()
    yearly_results = []

    for yr in distinct_years:
        yr_data = data[data["연도"] == yr].copy()
        result = []
        base_map = {"ATTACK": w_att, "MIDFIELDER": w_mid, "DEFENDER": w_def}

        for group, base in base_map.items():
            sub = yr_data[yr_data["POSITION_GROUP"] == group].copy()
            if len(sub) == 0: continue
            weights = normalize_weights(convert_weights_by_mode(base, data_mode, sub))
            sub = add_zscores(sub, weights)
            sub["포지션"] = sub.apply(lambda r: determine_detailed_position(r, data_mode), axis=1)
            sub[["수식어", "복합_스타일"]] = sub.apply(lambda r: pd.Series(assign_composite_roles(r, data_mode)), axis=1)
            sub = calculate_score(sub, weights)
            sub = analyze_strength_weakness(sub, weights, group)
            sub["USED_FEATURES"] = ", ".join(weights.keys())

            # Use only known match metrics, excluding identifiers and ranks.
            known = set(feature_korean) | set(per90_map.values())
            if data_mode == "90분당 데이터":
                features = [c for c in sub.columns if c in known and
                            (c.startswith("Per_") or c.startswith("PCNT_") or
                             c in {"AIR_CHALLENGE_PCNT", "Teckle_Won_Rate"})]
            else:
                features = [c for c in sub.columns if c in known and not c.startswith("Per_")]
            scaled = standardize_metrics(sub, features)
            for col in scaled:
                sub["ML_" + col] = scaled[col]
            n_clst = min({"ATTACK": 5, "MIDFIELDER": 6, "DEFENDER": 4}[group], len(sub))
            if not scaled.empty and len(scaled.columns):
                n_clst = min(n_clst, len(scaled.drop_duplicates()))
                sub["CLUSTER"] = KMeans(n_clusters=n_clst, random_state=42, n_init=10).fit_predict(scaled)
            else:
                sub["CLUSTER"] = 0

            result.append(sub)
        if result:
            yearly_results.append(pd.concat(result, ignore_index=True))

    return pd.concat(yearly_results, ignore_index=True) if yearly_results else pd.DataFrame()

# =========================================================
# 4. 사용자 UI 구성부
# =========================================================
# =========================================================
# 4. 사용자 UI 구성부
# =========================================================


# =========================================================
# 4. 사용자 UI 구성부
# =========================================================

uploaded_file = st.file_uploader(
    "데이터 파일 업로드",
    type=["xlsx", "xls", "csv"]
)

use_sample = st.checkbox("가상 샘플 데이터로 실행", value=uploaded_file is None)
if use_sample:
    st.caption("현재 가상 데이터를 분석 중입니다. 실제 업로드 파일을 분석하려면 가상 샘플 실행을 해제하세요.")
st.caption("실제 데이터의 출처·재배포 권한은 미확인입니다. 공개 시 가상 샘플 또는 권한을 확인한 파일을 사용하세요.")

if uploaded_file is None and not use_sample:
    st.info("데이터 파일을 업로드하세요.")
    st.stop()

try:
    df = make_sample_data() if use_sample else load_data(uploaded_file)
    st.success("파일을 성공적으로 불러왔습니다.")

except (
    ValueError,
    OSError,
    KeyError,
    TypeError,
    ImportError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
) as e:
    st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

required_cols = ["연도", "팀명", "선수명", "포지션", "출전시간"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(
        f"파일에 필요한 열이 없습니다: {', '.join(missing_cols)}"
    )
    st.stop()

df["POSITION_GROUP"] = df["포지션"].apply(classify_position_group)

metadata_cols = ["연도", "팀명", "선수명", "포지션", "출전시간", "POSITION_GROUP", "WORK_TIME",
                 "RAW_WEIGHTED_SCORE", "SCOUTING_SCORE", "GRADE", "수식어", "복합_스타일",
                 "강점", "약점", "USED_FEATURES", "CLUSTER"]
all_features = [c for c in df.columns if c not in metadata_cols and not c.endswith('_Z') and not c.startswith('Per_')]

st.sidebar.header("분석 조건")
data_mode = st.sidebar.radio("데이터 표출 모드", ["누적 데이터", "90분당 데이터"])
min_minutes = st.sidebar.slider("최소 출전시간", 0, 3000, 300, 50)

years = [y for y in df["연도"].unique().tolist() if y] if "연도" in df.columns else []
selected_year = st.sidebar.selectbox("시즌 선택", ["전체"] + sorted(years, reverse=True))
teams = [t for t in df["팀명"].unique().tolist() if t]
selected_team = st.sidebar.selectbox("구단 선택", ["전체"] + sorted(teams))

with st.sidebar.expander("🛠️ 포지션 대분류별 지표 가중치 조절", expanded=False):
    st.caption("기본 지표 + 평가 제외 지표 직접 추가 가능 (합계 1.0 자동 정규화)")
    tabs = st.tabs(["ATTACK", "MIDFIELDER", "DEFENDER"])
    cfg = [("att", base_attack_weights), ("mid", base_midfielder_weights), ("def", base_defender_weights)]
    customs = {}
    for i, (pfx, base) in enumerate(cfg):
        with tabs[i]:
            cw = {}
            st.markdown("##### 🔹 기본 평가 지표")
            for k, v in base.items():
                cw[k] = st.slider(feature_korean.get(k, k), 0.0, 1.0, float(v), 0.01, key=f"{pfx}_{k}")
            st.markdown("---")
            st.markdown("##### ➕ 평가 제외 지표 추가")
            addable = sorted([c for c in all_features if c not in base])
            extra = st.multiselect(f"추가할 지표 선택 ({pfx.upper()})", options=addable,
                                   format_func=lambda x: f"{feature_korean.get(x, x)} ({x})", key=f"extra_{pfx}_sel")
            for k in extra:
                cw[k] = st.slider(f"🌟 {feature_korean.get(k, k)}", 0.0, 1.0, 0.05, 0.01, key=f"{pfx}_extra_{k}")
            customs[pfx] = cw

attack_weights_custom = customs["att"]
midfielder_weights_custom = customs["mid"]
defender_weights_custom = customs["def"]

# 필터링 순서 최적화: 출전시간 통과자들을 대상으로 연도별/포지션별 독립 모형 실행
working_df = df[df["출전시간"] >= min_minutes].copy()

model_df = apply_model(working_df, data_mode, attack_weights_custom, midfielder_weights_custom, defender_weights_custom)
final_df = model_df.copy()
if len(final_df) == 0:
    st.warning("분석 가능한 데이터가 없습니다."); st.stop()

# 화면 출력 전용 최종 필터링 작동
if "연도" in final_df.columns and selected_year != "전체":
    final_df = final_df[final_df["연도"] == selected_year]
if selected_team != "전체":
    final_df = final_df[final_df["팀명"] == selected_team]
if len(final_df) == 0:
    st.warning(f"선택한 조건('{selected_year}'시즌 / '{selected_team}')의 데이터가 존재하지 않습니다."); st.stop()

all_pos = [p for p in final_df["포지션"].unique().tolist() if p and str(p).lower() != 'nan']
selected_position = st.sidebar.selectbox("포지션 선택 (대분류/세부)",
    ["전체", "ATTACK (그룹)", "MIDFIELDER (그룹)", "DEFENDER (그룹)"] + sorted(all_pos))

available_modifiers = ["전체"] + sorted([m for m in final_df["수식어"].unique().tolist() if m])
selected_modifier = st.sidebar.selectbox("🎯 선수 수식어(플레이스타일) 선택", available_modifiers)
search_name = st.sidebar.text_input("선수명 검색", "")

if selected_position != "전체":
    if " (그룹)" in selected_position:
        final_df = final_df[final_df["POSITION_GROUP"] == selected_position.replace(" (그룹)", "")]
    else:
        final_df = final_df[final_df["포지션"] == selected_position]
if selected_modifier != "전체":
    final_df = final_df[final_df["수식어"] == selected_modifier]

display_df = final_df.copy()
if search_name:
    display_df = display_df[display_df["선수명"].astype(str).str.contains(search_name, na=False)]
if len(display_df) == 0:
    st.error("❌ 조건에 일치하는 선수가 없습니다."); st.stop()

display_df = display_df.sort_values(by="SCOUTING_SCORE", ascending=False)
player_options = (display_df["선수명"].astype(str) + " / " + display_df["팀명"].astype(str) + " / " +
                  display_df["포지션"].astype(str) + " / " + display_df["수식어"].astype(str) + " / " +
                  display_df["연도"].astype(str))

selected_player_label = st.selectbox("상세 분석 선수 선택", player_options.tolist())
selected_idx = player_options[player_options == selected_player_label].index[0]
player = display_df.loc[selected_idx]

# =========================================================
# 5. 상단 지표 요약 스코어보드
# =========================================================
gv = str(player["GRADE"]).strip()
grade_map = {
    "S+": "<span style='color:#FFD700;font-weight:bold;font-size:20px;'>👑 S+ 등급</span>",
    "S": "<span style='color:#D4AF37;font-weight:bold;font-size:20px;'>🥇 S 등급</span>",
    "A+": "<span style='color:#C0C0C0;font-weight:bold;font-size:20px;'>🥈 A+ 등급</span>",
    "A": "<span style='color:#94A3B8;font-weight:bold;font-size:20px;'>✨ A 등급</span>",
    "B+": "<span style='color:#CD7F32;font-weight:bold;font-size:20px;'>🥉 B+ 등급</span>",
    "B": "<span style='color:#B45309;font-weight:bold;font-size:20px;'>⚡ B 등급</span>",
    "C+": "<span style='color:#808080;font-weight:bold;font-size:20px;'>🪙 C+ 등급</span>",
    "C": "<span style='color:#6B7280;font-weight:bold;font-size:20px;'>🛡️ C 등급</span>",
    "D+": "<span style='color:#9CA3AF;font-weight:bold;font-size:20px;'>🎯 D+ 등급</span>",
    "D": "<span style='color:#E5E7EB;font-weight:bold;font-size:20px;'>📋 D 등급</span>"
}
grade_html = grade_map.get(gv, f"<span style='color:#808080;font-weight:bold;font-size:20px;'>🪙 {gv} 등급</span>")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("선수 성명", f"{player['선수명']}")
c2.metric("팀명", player["팀명"])
c3.metric("판단 포지션", player["포지션"])
c4.metric("출전시간", f"{int(player['출전시간'])}분")
c5.metric("상대평가 점수", f"{player['SCOUTING_SCORE']} 점")
with c6:
    st.markdown("<label style='font-size:14px;color:gray;'>선수 등급</label>", unsafe_allow_html=True)
    st.markdown(grade_html, unsafe_allow_html=True)

st.markdown(f"""
    <div style="background-color:#2E1065;padding:12px 20px;border-radius:8px;margin:15px 0;">
        <span style="color:#A78BFA;font-size:14px;font-weight:bold;opacity:0.9;">⚽ AI 복합 플레이스타일 분석 (해당 시즌 기준 독립 산출)</span><br>
        <span style="color:#FFFFFF;font-size:20px;font-weight:bold;letter-spacing:0.5px;">
            {str(player['복합_스타일']).strip()} — {player['선수명']}
        </span>
    </div>
""", unsafe_allow_html=True)

sc1, sc2 = st.columns(2)
with sc1:
    st.markdown(f"""
        <div style="background-color:#064E3B;padding:12px 18px;border-radius:8px;">
            <span style="color:#6EE7B7;font-size:14px;font-weight:bold;">💪 강점 ({player['연도']}시즌 평균 150% 이상)</span><br>
            <span style="color:#FFFFFF;font-size:16px;">{player['강점']}</span>
        </div>
    """, unsafe_allow_html=True)
with sc2:
    st.markdown(f"""
        <div style="background-color:#7F1D1D;padding:12px 18px;border-radius:8px;">
            <span style="color:#FCA5A5;font-size:14px;font-weight:bold;">⚠️ 약점 ({player['연도']}시즌 평균 70% 이하)</span><br>
            <span style="color:#FFFFFF;font-size:16px;">{player['약점']}</span>
        </div>
    """, unsafe_allow_html=True)

st.write("---")

# =========================================================
# 6. 단독 선수의 Z-score 레이더 차트 및 원점수 테이블
# =========================================================
group = player["POSITION_GROUP"]
base_weights = {"ATTACK": attack_weights_custom, "MIDFIELDER": midfielder_weights_custom, "DEFENDER": defender_weights_custom}[group]
used_weights = normalize_weights(convert_weights_by_mode(base_weights, data_mode, final_df))
sorted_uw = sorted(used_weights.items(), key=lambda i: i[1], reverse=True)
radar_features = [f for f, _ in sorted_uw][:10]

target_miss = "Per_BALL_MISS" if data_mode == "90분당 데이터" else "BALL_MISS"
if group == "DEFENDER" and target_miss in used_weights and target_miss not in radar_features:
    radar_features[-1] = target_miss

z_values, raw_table = [], []
for f in radar_features:
    zc = f + "_Z"
    oz = float(player[zc]) if zc in player.index else 0
    is_neg = f in negative_features or f.replace("Per_", "") in negative_features
    rz = -oz if is_neg else oz
    rv = float(player[f]) if f in player.index else 0
    z_values.append(rz)
    raw_table.append({
        "지표": feature_korean.get(f, f), "원점수": round(rv, 2),
        "시즌 보정 Z-score": round(rz, 2), "가중치": round(used_weights.get(f, 0.0), 4),
        "반영": "감점" if is_neg else "가점"
    })

left, right = st.columns([1.2, 1])
with left:
    st.subheader(f"🎯 {player['선수명']} Z-score 평가지표 레이더 차트")
    st.caption(f"※ **{player['연도']} 시즌** 평균(0) 대비 해당 선수의 상대적 위치 분석 도표입니다.")
    theta = [feature_korean.get(f, f) for f in radar_features]
    mx, mn = (max(z_values), min(z_values)) if z_values else (2, -2)
    pmax, pmin = max(2.5, mx + 0.5), min(-2.5, mn - 0.5)
    if not z_values or not theta:
        st.warning(
            "레이더 차트를 그릴 지표가 없습니다. 업로드 파일의 지표 열 이름과 "
            "선택한 포지션의 가중치 설정을 확인해 주세요."
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=z_values + [z_values[0]], theta=theta + [theta[0]],
            fill="toself", name=str(player["선수명"]), line={'color': '#1E40AF', 'width': 3},
            fillcolor="rgba(30,64,175,0.25)"))
        fig.add_trace(go.Scatterpolar(r=[0]*(len(radar_features)+1), theta=theta + [theta[0]],
            mode="lines", name="해당 시즌 평균 (0)", line={'color': '#EF4444', 'width': 3, 'dash': 'dash'}))
        fig.update_layout(
            polar={
                'radialaxis': {'visible': True, 'range': [pmin, pmax]},
                'angularaxis': {'tickfont': {'size': 12}},
            },
            height=600,
            margin={'l': 100, 'r': 100, 't': 40, 'b': 40},
            showlegend=True,
            legend={
                'orientation': 'h', 'yanchor': 'bottom', 'y': -0.15,
                'xanchor': 'center', 'x': 0.5,
            },
        )
        st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("📊 원점수 테이블")
    st.caption(f"현재 표출 모드: {data_mode}")
    st.dataframe(pd.DataFrame(raw_table), width="stretch")

# =========================================================
# [새로 추가] 6-2. 선택 선수의 연도별(시즌별) 스카우팅 히스토리 트렌드 분석
# =========================================================
st.write("---")
st.subheader(f"📈 {player['선수명']} 선수 시즌별 변화 추적")
st.caption("선택된 선수의 과거 시즌 데이터를 추적하여 AI 합산 스코어와 주요 지표가 어떻게 변화했는지 분석합니다.")

# 화면 필터 전 모델 결과에서 이력을 조회합니다.
player_history = get_player_history(model_df, player["선수명"])
st.caption(f"시즌·구단·포지션·스타일 화면 필터와 무관한 이력입니다. 최소 출전시간 {min_minutes}분 미만과 분석 대상 외 포지션은 점수 이력에서 제외됩니다. 동명이인은 이름만으로 구분되지 않습니다. 시즌별 점수는 각 시즌 비교집단 내 상대 위치입니다.")
with st.expander("원본의 전체 시즌 기록 (최소 출전시간 적용 전)"):
    raw_history = get_player_history(df, player["선수명"])
    st.dataframe(raw_history[["연도", "팀명", "선수명", "포지션", "출전시간"]], width="stretch")

if len(player_history) > 1:
    h_col1, h_col2 = st.columns([1.2, 1])

    with h_col1:
        st.markdown("**시즌별 AI 합산 스코어(SCOUTING_SCORE) 변화 추이**")
        # 선형 차트 객체 생성 및 등급 대신 AI 합산 스코어로 수치 매핑
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=player_history["연도"],
            y=player_history["SCOUTING_SCORE"],
            mode="lines+markers+text",
            text=player_history["SCOUTING_SCORE"].astype(str) + "점",  # 등급(GRADE) 대신 AI 스코어로 표현
            textposition="top center",
            line={'color': '#2E1065', 'width': 3},
            marker={'size': 10, 'color': '#A78BFA'},
            name="AI 합산 스코어"
        ))
        fig_trend.update_layout(
            height=350,
            margin={'l': 40, 'r': 40, 't': 20, 'b': 40},
            xaxis={'title': '시즌 (연도)', 'type': 'category'},
            yaxis={'title': '점수 (Score)', 'range': [0, 100]}
        )
        st.plotly_chart(fig_trend, width="stretch")

    with h_col2:
        st.markdown("**시즌별 요약 테이블**")
        history_table = player_history[["연도", "팀명", "포지션", "출전시간", "SCOUTING_SCORE", "GRADE", "수식어"]].reset_index(drop=True)
        history_table = history_table.rename(columns={"SCOUTING_SCORE": "AI 스코어", "GRADE": "등급"})
        st.dataframe(history_table, width="stretch")

    # 다중 시즌 레이더 차트 오버레이 비교
    st.markdown("**시즌별 레이더 차트 겹쳐보기 (역량 변화 대조)**")
    fig_hist_radar = go.Figure()

    # 기준선(시즌 평균 0) 추가
    fig_hist_radar.add_trace(go.Scatterpolar(
        r=[0] * (len(radar_features) + 1), theta=theta + [theta[0]],
        mode="lines", name="해당 시즌 평균 (0)", line={'color': '#EF4444', 'width': 2, 'dash': 'dash'}
    ))

    max_hz_list, min_hz_list = [2.5], [-2.5]
    for _, h_row in player_history.iterrows():
        h_z_values = []
        for f in radar_features:
            zc = f + "_Z"
            oz = float(h_row[zc]) if zc in h_row.index else 0
            is_neg = f in negative_features or f.replace("Per_", "") in negative_features
            rz = -oz if is_neg else oz
            h_z_values.append(rz)

        max_hz_list.append(max(h_z_values))
        min_hz_list.append(min(h_z_values))

        fig_hist_radar.add_trace(go.Scatterpolar(
            r=h_z_values + [h_z_values[0]], theta=theta + [theta[0]],
            fill="none", name=f"{h_row['연도']} 시즌 ({h_row['팀명']}) [AI 스코어: {h_row['SCOUTING_SCORE']}점]",
            line={'width': 2.5}
        ))

    fig_hist_radar.update_layout(
        polar={
            'radialaxis': {
                'visible': True,
                'range': [min(min_hz_list) - 0.5, max(max_hz_list) + 0.5],
            },
            'angularaxis': {'tickfont': {'size': 12}},
        },
        height=500,
        margin={'l': 80, 'r': 80, 't': 20, 'b': 40},
        showlegend=True,
        legend={
            'orientation': 'h', 'yanchor': 'bottom', 'y': -0.15,
            'xanchor': 'center', 'x': 0.5,
        },
    )
    st.plotly_chart(fig_hist_radar, width="stretch")

else:
    st.info("현재 분석 조건을 통과한 이력 행이 1개입니다. 원본 전체 시즌 기록과 최소 출전시간 조건을 확인하세요.")

# =========================================================
# 7. 다자간 역량 대조 레이더 차트 분석 섹션
# =========================================================
st.write("---")
st.subheader("선수비교분석")
st.caption("비교를 원하는 세부 포지션을 필터링한 후 여러 선수들을 다중 선택하여 비교하는 영역입니다. (각 선수의 수치는 해당 시즌의 평균 기준 분포입니다.)")

comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    radar_group_filter = st.selectbox(
        "1단계: 비교군 포지션 대분류 필터",
        ["전체", "ATTACK", "MIDFIELDER", "DEFENDER"], key="comp_grp"
    )

with comp_col2:
    if radar_group_filter == "ATTACK":
        sub_pos_options = ["전체", "ST", "L/RW"]
    elif radar_group_filter == "MIDFIELDER":
        sub_pos_options = ["전체", "CAM", "CM", "CDM"]
    elif radar_group_filter == "DEFENDER":
        sub_pos_options = ["전체", "CB", "L/RB"]
    else:
        sub_pos_options = ["전체"] + sorted([p for p in final_df["포지션"].unique().tolist() if p])

    radar_sub_filter = st.selectbox("2단계: 비교군 세부 포지션 필터", sub_pos_options, key="comp_sub")

radar_cand_df = final_df.copy()
if radar_group_filter != "전체":
    radar_cand_df = radar_cand_df[radar_cand_df["POSITION_GROUP"] == radar_group_filter]
if radar_sub_filter != "전체":
    radar_cand_df = radar_cand_df[radar_cand_df["포지션"] == radar_sub_filter]

radar_cand_df["PLAYER_LABEL"] = (
    radar_cand_df["선수명"].astype(str) + " (" +
    radar_cand_df["팀명"].astype(str) + " / " +
    radar_cand_df["포지션"].astype(str) + " / " +
    radar_cand_df["연도"].astype(str) + ")"
)

default_label = f"{player['선수명']} ({player['팀명']} / {player['포지션']} / {player['연도']})"
all_cand_labels = radar_cand_df["PLAYER_LABEL"].tolist()

if default_label not in all_cand_labels:
    all_cand_labels = [default_label] + all_cand_labels

selected_comparison_players = st.multiselect(
    "🔍 레이더 차트에 중첩하여 비교할 선수들을 다중 선택하세요",
    options=all_cand_labels,
    default=[default_label],
    key="comp_multiselect"
)

left_comp, right_comp = st.columns([1.2, 1])

with left_comp:
    # 🌟 대분류 필터링 상태에 따라 독립 가중치 매핑 구조 추적 및 세부 포지션별 지표 가중치 추출 🌟
    active_grp = radar_group_filter if radar_group_filter != "전체" else player["POSITION_GROUP"]
    comp_base_weights = {"ATTACK": attack_weights_custom, "MIDFIELDER": midfielder_weights_custom, "DEFENDER": defender_weights_custom}[active_grp]
    comp_used_weights = normalize_weights(convert_weights_by_mode(comp_base_weights, data_mode, final_df))
    comp_sorted_uw = sorted(comp_used_weights.items(), key=lambda i: i[1], reverse=True)
    comp_radar_features = [f for f, _ in comp_sorted_uw][:10]

    # 수비수 예외처리 매핑 동기화
    comp_target_miss = "Per_BALL_MISS" if data_mode == "90분당 데이터" else "BALL_MISS"
    if active_grp == "DEFENDER" and comp_target_miss in comp_used_weights and comp_target_miss not in comp_radar_features:
        comp_radar_features[-1] = comp_target_miss

    comp_theta = [feature_korean.get(f, f) for f in comp_radar_features]

    fig_radar = go.Figure()

    # 🌟 요구사항 1: 평균 선(go.Scatterpolar 내부의 평균 데이터라인 트레이스) 제거 완료 🌟
    max_z_list, min_z_list = [2.5], [-2.5]

    for p_label in selected_comparison_players:
        p_match = final_df[
            (final_df["선수명"].astype(str) + " (" +
             final_df["팀명"].astype(str) + " / " +
             final_df["포지션"].astype(str) + " / " +
             final_df["연도"].astype(str) + ")") == p_label
        ]

        if len(p_match) == 0:
            continue

        p_row = p_match.iloc[0]
        p_z_values = []

        for f in comp_radar_features:
            zc = f + "_Z"
            oz = float(p_row[zc]) if zc in p_row.index else 0
            is_neg = f in negative_features or f.replace("Per_", "") in negative_features
            rz = -oz if is_neg else oz
            p_z_values.append(rz)

        max_z_list.append(max(p_z_values))
        min_z_list.append(min(p_z_values))

        # 🌟 요구사항 2: 각 필터링 포지션의 동적 매핑 연산을 통해 배열 마지막에 첫 인덱스([0])를 명시적으로 패딩하여 선이 온전하게 닫히도록 마감 처리 완료 🌟
        fig_radar.add_trace(go.Scatterpolar(
            r=p_z_values + [p_z_values[0]], theta=comp_theta + [comp_theta[0]],
            fill="toself", name=f"{p_row['선수명']} ({p_row['연도']} / {p_row['팀명']})",
            line={'width': 2.5}
        ))

    pmax = max(max_z_list) + 0.5
    pmin = min(min_z_list) - 0.5

    fig_radar.update_layout(
        polar={
            'radialaxis': {'visible': True, 'range': [pmin, pmax], 'title': 'Z-Score'},
            'angularaxis': {'tickfont': {'size': 12}},
        },
        height=600,
        margin={'l': 80, 'r': 80, 't': 40, 'b': 40},
        showlegend=True,
        legend={
            'orientation': 'h', 'yanchor': 'bottom', 'y': -0.15,
            'xanchor': 'center', 'x': 0.5,
        },
    )
    st.plotly_chart(fig_radar, width="stretch")

with right_comp:
    st.subheader("📋 선택 선수 원점수 대조 평점표")
    st.caption(f"현재 데이터 표출 기준: {data_mode}")

    table_records = []
    for f in comp_radar_features:
        row_dict = {"평가 지표": feature_korean.get(f, f)}
        for p_label in selected_comparison_players:
            p_match = final_df[
                (final_df["선수명"].astype(str) + " (" +
                 final_df["팀명"].astype(str) + " / " +
                 final_df["포지션"].astype(str) + " / " +
                 final_df["연도"].astype(str) + ")") == p_label
            ]
            if len(p_match) > 0:
                p_row = p_match.iloc[0]
                row_dict[f"{p_row['선수명']}({p_row['연도']})"] = round(float(p_row[f]), 2)
        table_records.append(row_dict)

    st.dataframe(pd.DataFrame(table_records), width="stretch")

st.write("---")

# =========================================================
# 8. 머신러닝 하이브리드 유사 대체 선수 추천 시스템
# =========================================================
st.subheader("유사 대체 선수 TOP 5")
st.caption(f"💡 [AI 엔진 분석 결과] 선택 선수 해당 시즌 머신러닝 전술 스타일 코드: **{player['CLUSTER']}번 군집**")

st.caption("동일 시즌·대분류에서 표준화한 지표를 비교합니다. 화면 필터 안의 같은 시즌·세부 포지션에서 같은 군집을 우선합니다. 코사인 유사도는 -1~1이며 적합 확률이 아닙니다. 실전 추천 타당성은 아직 검증되지 않았습니다.")
top5 = recommend_players(final_df, player)
if not top5.empty:
    cols = ["연도", "팀명", "선수명", "포지션", "수식어", "출전시간", "SCOUTING_SCORE", "GRADE", "코사인 유사도"]
    st.dataframe(top5[cols], width="stretch")
else:
    st.info("같은 시즌·포지션의 유효한 비교 대상 또는 변동이 있는 지표가 부족합니다.")

st.write("---")

# =========================================================
# 9. 전체 스카우팅 랭킹 리스트
# =========================================================
st.subheader(f"📋 스카우팅 랭킹 리스트 ({selected_year if selected_year!='전체' else '전체'} 시즌 / {selected_team if selected_team!='전체' else '전체'})")
rank_cols = ["연도", "팀명", "선수명", "포지션", "수식어", "강점", "약점", "출전시간", "SCOUTING_SCORE", "GRADE"]

final_df["출전시간"] = final_df["출전시간"].astype(int)

st.dataframe(final_df[rank_cols].sort_values(by="SCOUTING_SCORE", ascending=False).reset_index(drop=True),
             width="stretch")