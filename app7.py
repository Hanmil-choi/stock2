import streamlit as st
# --- Safe initialization for `custom_indicator_expr` ---
if 'custom_indicator_expr' not in globals():
    custom_indicator_expr = ""
# --------------------------------------------------------

import pandas as pd
import os
from glob import glob
import datetime as dt
import numpy as np

def find_column(df, target_names):
    for col in df.columns:
        if col.strip().lower() in [name.lower() for name in target_names]:
            return col
    return None

def calculate_returns(df, date_col, close_col):
    """상승률을 계산하는 함수"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # 2일, 20일, 60일 상승률 계산
    df['return_2d'] = df[close_col].pct_change(2) * 100
    df['return_20d'] = df[close_col].pct_change(20) * 100
    df['return_60d'] = df[close_col].pct_change(60) * 100
    
    return df

def calculate_relative_momentum(df_stock, df_benchmark, date_col, close_col, periods=[20, 60, 120]):
    """KODEX 200에 대한 상대 모멘텀을 계산하는 함수"""
    df = df_stock.copy()
    df_bm = df_benchmark.copy()
    
    # 날짜 컬럼 찾기
    bm_date_col = find_column(df_bm, ['거래일자', 'date', 'Date', '날짜'])
    bm_close_col = find_column(df_bm, ['종가', 'close', 'Close', '종가'])
    
    # 날짜 형식 통일 (YYYYMMDD 형식으로 파싱)
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    df_bm[bm_date_col] = pd.to_datetime(df_bm[bm_date_col], format='%Y%m%d')
    
    # 날짜 기준으로 병합
    merged = pd.merge(
        df[[date_col, close_col]], 
        df_bm[[bm_date_col, bm_close_col]].rename(columns={bm_date_col: date_col, bm_close_col: "bm_close"}), 
        on=date_col, 
        how="inner"
    )
    
    # 각 기간별 상대 모멘텀 계산
    for period in periods:
        merged[f"rel_mom_{period}"] = (
            (merged[close_col] / merged[close_col].shift(period)) /
            (merged["bm_close"] / merged["bm_close"].shift(period)) - 1
        ) * 100  # 백분율로 변환
    
    # 원본 데이터프레임에 상대 모멘텀 컬럼 추가 (날짜 기준으로 매핑)
    result_df = df.copy()
    for period in periods:
        # 날짜를 키로 사용하여 매핑
        merged_subset = merged[[date_col, f"rel_mom_{period}"]].set_index(date_col)
        result_df[f"rel_mom_{period}"] = result_df[date_col].map(merged_subset[f"rel_mom_{period}"])
    
    return result_df

def calculate_52week_high_low(df, date_col, close_col, high_col, low_col):
    """52주 고점/저점을 계산하는 함수"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # 52주(약 252거래일) 고점/저점 계산
    df['high_52w'] = df[high_col].rolling(window=252, min_periods=1).max()
    df['low_52w'] = df[low_col].rolling(window=252, min_periods=1).min()
    
    # 현재가 대비 52주 고점/저점 비율
    df['high_52w_ratio'] = (df[close_col] / df['high_52w']) * 100
    df['low_52w_ratio'] = (df[close_col] / df['low_52w']) * 100
    
    return df

def calculate_sma(df, close_col, periods=[5, 10, 20, 60, 120]):
    """단순이동평균선을 계산하는 함수"""
    df = df.copy()
    for period in periods:
        df[f'sma{period}'] = df[close_col].rolling(window=period, min_periods=1).mean()
    return df
    
def calculate_ema(df, close_col, periods=[12, 26]):
    """지수이동평균선을 계산하는 함수"""
    df = df.copy()
    for period in periods:
        df[f'ema{period}'] = df[close_col].ewm(span=period, adjust=False).mean()
    return df

def calculate_rsi(df, close_col, period=14):
    """RSI를 계산하는 함수"""
    df = df.copy()
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, close_col, fast=12, slow=26, signal=9):
    """MACD를 계산하는 함수"""
    df = df.copy()
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df

def calculate_volume_indicators(df, volume_col, close_col):
    """거래량 지표를 계산하는 함수"""
    df = df.copy()
    
    # 거래량 이동평균
    df['volume_sma5'] = df[volume_col].rolling(window=5, min_periods=1).mean()
    df['volume_sma20'] = df[volume_col].rolling(window=20, min_periods=1).mean()
    
    # 거래량 비율 (현재 거래량 / 20일 평균 거래량)
    df['volume_ratio'] = df[volume_col] / df['volume_sma20']
    
    return df

def calculate_bollinger_bands(df, close_col, period=20, std_dev=2):
    """볼린저 밴드를 계산하는 함수"""
    df = df.copy()
    df['bb_middle'] = df[close_col].rolling(window=period, min_periods=1).mean()
    bb_std = df[close_col].rolling(window=period, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
    return df

def calculate_atr(df, high_col, low_col, close_col, period=14):
    """ATR(Average True Range)를 계산하는 함수"""
    df = df.copy()
    high_low = df[high_col] - df[low_col]
    high_close = abs(df[high_col] - df[close_col].shift())
    low_close = abs(df[low_col] - df[close_col].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=period, min_periods=1).mean()
    return df

def calculate_volatility(df, close_col, periods=[20, 60]):
    """변동성을 계산하는 함수"""
    df = df.copy()
    
    for period in periods:
        # 수익률 계산
        returns = df[close_col].pct_change()
        
        # 변동성 (표준편차)
        df[f'volatility_{period}'] = returns.rolling(window=period, min_periods=1).std() * 100
        
        # 연율화된 변동성 (252거래일 기준)
        df[f'volatility_annualized_{period}'] = returns.rolling(window=period, min_periods=1).std() * (252 ** 0.5) * 100
    
    return df

def calculate_parkinson_volatility(df, high_col, low_col, period=20):
    """Parkinson 변동성을 계산하는 함수 (고가-저가 기반)"""
    df = df.copy()
    
    # 고가/저가 비율의 로그
    log_hl_ratio = np.log(df[high_col] / df[low_col])
    
    # Parkinson 변동성 계산
    df['parkinson_volatility'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        log_hl_ratio.rolling(window=period, min_periods=1).mean()
    ) * 100
    
    return df

def calculate_garman_klass_volatility(df, open_col, high_col, low_col, close_col, period=20):
    """Garman-Klass 변동성을 계산하는 함수 (OHLC 기반)"""
    df = df.copy()
    
    # Garman-Klass 변동성 계산
    log_hl = np.log(df[high_col] / df[low_col])
    log_co = np.log(df[close_col] / df[open_col])
    
    volatility = np.sqrt(
        (0.5 * log_hl**2) - ((2*np.log(2) - 1) * log_co**2)
    )
    
    df['garman_klass_volatility'] = volatility.rolling(window=period, min_periods=1).mean() * 100
    
    return df

def calculate_true_range_volatility(df, high_col, low_col, close_col, period=20):
    """True Range 기반 변동성을 계산하는 함수"""
    df = df.copy()
    
    # True Range 계산
    high_low = df[high_col] - df[low_col]
    high_close = abs(df[high_col] - df[close_col].shift())
    low_close = abs(df[low_col] - df[close_col].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # True Range 기반 변동성
    df['true_range_volatility'] = (true_range / df[close_col]).rolling(window=period, min_periods=1).mean() * 100
    
    return df

def calculate_sharpe_ratio(df, close_col, periods=[20, 60, 120], risk_free_rate=0.02):
    """Sharpe 지수를 계산하는 함수"""
    df = df.copy()
    
    # 수익률 계산
    returns = df[close_col].pct_change()
    
    for period in periods:
        # 기간별 평균 수익률 (연율화)
        mean_return = returns.rolling(window=period, min_periods=1).mean() * 252
        
        # 기간별 수익률 표준편차 (연율화)
        std_return = returns.rolling(window=period, min_periods=1).std() * (252 ** 0.5)
        
        # Sharpe 지수 계산: (수익률 - 무위험수익률) / 표준편차
        df[f'sharpe_ratio_{period}'] = (mean_return - risk_free_rate) / (std_return + 1e-8)
    
    return df

def calculate_sortino_ratio(df, close_col, periods=[20, 60, 120], risk_free_rate=0.02):
    """Sortino 지수를 계산하는 함수 (하방 위험만 고려)"""
    df = df.copy()
    
    # 수익률 계산
    returns = df[close_col].pct_change()
    
    for period in periods:
        # 기간별 평균 수익률 (연율화)
        mean_return = returns.rolling(window=period, min_periods=1).mean() * 252
        
        # 하방 수익률만 추출 (음수 수익률)
        downside_returns = returns.where(returns < 0, 0)
        
        # 하방 표준편차 (연율화)
        downside_std = downside_returns.rolling(window=period, min_periods=1).std() * (252 ** 0.5)
        
        # Sortino 지수 계산: (수익률 - 무위험수익률) / 하방표준편차
        df[f'sortino_ratio_{period}'] = (mean_return - risk_free_rate) / (downside_std + 1e-8)
    
    return df

def calculate_calmar_ratio(df, close_col, periods=[20, 60, 120], risk_free_rate=0.02):
    """Calmar 지수를 계산하는 함수 (최대 낙폭 대비 수익률)"""
    df = df.copy()
    
    # 수익률 계산
    returns = df[close_col].pct_change()
    
    for period in periods:
        # 누적 수익률 계산
        cumulative_returns = (1 + returns).rolling(window=period, min_periods=1).apply(np.prod) - 1
        
        # 연율화된 수익률
        annualized_return = ((1 + cumulative_returns) ** (252 / period)) - 1
        
        # 최대 낙폭 계산
        rolling_max = cumulative_returns.rolling(window=period, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / (rolling_max + 1e-8)
        max_drawdown = drawdown.rolling(window=period, min_periods=1).min()
        
        # Calmar 지수 계산: (수익률 - 무위험수익률) / 최대낙폭
        df[f'calmar_ratio_{period}'] = (annualized_return - risk_free_rate) / (abs(max_drawdown) + 1e-8)
    
    return df

# ---- 유틸: 크로스 탐지
def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))

# ---- 안전한 상관관계 계산 (scipy 없이)
def safe_spearman_corr(x, y):
    """scipy 없이 Spearman 상관관계를 계산하는 함수"""
    try:
        # 결측치 제거
        valid_mask = ~(x.isna() | y.isna())
        if valid_mask.sum() < 2:
            return 0.0
            
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # 순위 계산
        x_rank = x_clean.rank()
        y_rank = y_clean.rank()
        
        # 평균 계산
        x_mean = x_rank.mean()
        y_mean = y_rank.mean()
        
        # 분자 계산: (x - x_mean) * (y - y_mean)의 합
        numerator = ((x_rank - x_mean) * (y_rank - y_mean)).sum()
        
        # 분모 계산: sqrt(sum((x - x_mean)^2) * sum((y - y_mean)^2))
        x_var = ((x_rank - x_mean) ** 2).sum()
        y_var = ((y_rank - y_mean) ** 2).sum()
        denominator = (x_var * y_var) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    except Exception as e:
        print(f"상관관계 계산 오류: {e}")
        return 0.0

# ---- 정규화
def winsorize(s, p_low=0.01, p_high=0.99):
    lo, hi = s.quantile(p_low), s.quantile(p_high)
    return s.clip(lo, hi)

# ---- 개선된 정규화 함수
def normalize(series, method="rank", winsor=None, clip=None, higher_is_better=True, mode=None):
    """개선된 정규화 함수 (지시문 기반)"""
    s = series.copy()
    
    # 결측치 처리
    s = s.fillna(0)
    
    if len(s) == 0:
        return pd.Series([50.0] * len(series), index=series.index)  # 중간값 반환
    
    # 단일 값인 경우 중간값 반환
    if len(s) == 1:
        return pd.Series([50.0], index=s.index)
    
    # 극단값 완화
    if winsor:
        lo, hi = s.quantile(winsor[0]), s.quantile(winsor[1])
        s = s.clip(lo, hi)
    
    # 정규화 방법
    if method == "minmax":
        rng = s.max() - s.min()
        out = (s - s.min()) / rng if rng != 0 else pd.Series(0.5, index=s.index)
    elif method == "zscore":
        std = s.std(ddof=0)
        out = (s - s.mean()) / (std if std else 1.0)
        # z는 -3~+3 정도가 자연스러움 → 0~100으로 매핑
        out = (out.clip(-3, 3) + 3) / 6
    elif method == "rank":
        out = s.rank(method="average", pct=True)
    else:
        out = s
    
    # 모드별 특별 처리
    if mode == "revert":
        # 극단 회귀 모드: 중간값이 좋음 (∩자 형태)
        out = 100 - abs(out - 50) * 2
        out = out.clip(0, 100)
    elif mode == "breakout":
        # 돌파 추종 모드: 극단값이 좋음 (선형)
        pass  # 기본 처리 유지
    
    if clip:
        out = out.clip(clip[0], clip[1])
    
    # 방향성 조정
    if not higher_is_better:
        out = 100 - out
    
    # 0~100 스케일
    out = (out - out.min()) / (out.max() - out.min() + 1e-12) * 100
    
    return out

# ---- 불리언/이벤트 점수
def eval_boolean_expr(df, expr):
    """컬럼명 매핑을 포함한 불리언 표현식 평가"""
    # 컬럼명 매핑
    column_mapping = {
        'close': '종가',
        'open': '시가', 
        'high': '고가',
        'low': '저가',
        'volume': '거래량',
        'date': '거래일자'
    }
    
    # 표현식에서 컬럼명 치환
    modified_expr = expr
    for eng_name, kor_name in column_mapping.items():
        if kor_name in df.columns:
            modified_expr = modified_expr.replace(eng_name, kor_name)
    
    try:
        return pd.eval(modified_expr, local_dict=df.to_dict(orient="series"))
    except Exception as e:
        print(f"Expression evaluation error: {e}")
        print(f"Original expr: {expr}")
        print(f"Modified expr: {modified_expr}")
        print(f"Available columns: {list(df.columns)}")
        return pd.Series([False] * len(df), index=df.index)

def days_since_event(mask):
    # True인 날이 이벤트 발생일. 이후 경과일 계산
    idx = mask.index
    last_day = -np.inf
    out = []
    for i, flag in enumerate(mask.values):
        if flag:
            last_day = i
        out.append(i - last_day if np.isfinite(last_day) else np.inf)
    return pd.Series(out, index=idx)

def apply_rules(df, rules):
    """룰을 적용하여 점수를 계산하는 함수"""
    rule_points = pd.Series(0.0, index=df.index)
    log_cols = {}
    
    for rule in rules:
        if rule["type"] == "boolean":
            m = eval_boolean_expr(df, rule["expr"])
            pts = np.where(m, rule["score"], 0.0)
        elif rule["type"] == "event_decay":
            ev = rule["event"]
            if ev.startswith("CROSSOVER("):
                a, b = ev[len("CROSSOVER("):-1].split(",")
                a, b = a.strip(), b.strip()
                m = crossover(df[a], df[b])
            elif ev.startswith("CROSSUNDER("):
                a, b = ev[len("CROSSUNDER("):-1].split(",")
                a, b = a.strip(), b.strip()
                m = crossunder(df[a], df[b])
            else:
                raise ValueError("Unknown event")
            t = days_since_event(m)
            decay = np.exp(-t / rule["tau"])
            pts = np.where(np.isfinite(t), rule["base"] * decay, 0.0)
        else:
            raise ValueError("Unknown rule type")
        rule_points += pts
        log_cols[rule["name"]] = pts
    
    # 빈 DataFrame을 반환할 때 인덱스 명시적 지정
    if log_cols:
        return rule_points, pd.DataFrame(log_cols, index=df.index)
    else:
        return rule_points, pd.DataFrame(index=df.index)

# ---- 개선된 스코어러
def score_frame(df, config):
    """개선된 스코어링 시스템"""
    # 단일 행인 경우 결측치 처리를 다르게 함
    if len(df) == 1:
        # 단일 행의 경우 dropna() 대신 결측치를 0으로 채움
        df = df.fillna(0)
    else:
        # 여러 행인 경우 기존 방식 사용
        df = df.ffill().dropna()
    
    if len(df) == 0:
        print("Warning: DataFrame is empty after processing")
        return pd.Series([0.0] * len(df), index=df.index), pd.DataFrame()
    
    # 1) 숫자형 피처 점수화
    num_cfg = config.get("numeric", {})
    num_log = {}
    total_num = pd.Series(0.0, index=df.index)
    
    for feat, spec in num_cfg.items():
        if feat not in df.columns:
            print(f"Feature {feat} not found in DataFrame")
            continue
            
        try:
            s = df[feat].astype(float)
            # 결측치가 있는 경우 0으로 채움
            s = s.fillna(0)
            
            # 모든 값이 0인 경우 건너뛰기
            if (s == 0).all():
                print(f"Feature {feat} has all zero values")
                continue
                
            vec = normalize(
                s,
                method=spec.get("norm", "rank"),
                winsor=spec.get("winsor", [0.02, 0.98]),
                clip=spec.get("clip"),
                higher_is_better=spec.get("higher_is_better", True),
                mode=spec.get("mode")  # 모드 지원 추가
            )
            w = float(spec.get("weight", 0))
            total_num += vec * w
            num_log[f"[N] {feat}"] = vec * w
        except Exception as e:
            print(f"Error processing numeric feature {feat}: {e}")
            continue

    # 2) 룰 점수화
    rule_points, rule_log = apply_rules(df, config.get("rules", []))

    # 3) 합산
    total = total_num + rule_points

    # 4) 필터 상한 적용
    cap_cfg = config.get("filters", {})
    if cap_cfg and cap_cfg.get("exprs"):
        pass_mask = np.ones(len(df), dtype=bool)
        for expr in cap_cfg["exprs"]:
            try:
                pass_mask &= eval_boolean_expr(df, expr).values
            except:
                pass_mask &= False
        
        cap_value = cap_cfg.get("cap_if_fail", 40.0)
        total = np.where(pass_mask, total, np.minimum(total, cap_value))
        total = pd.Series(total, index=df.index)

    # 5) 기여도 로그 생성
    contrib = pd.concat([pd.DataFrame(num_log), rule_log], axis=1)
    contrib["TOTAL"] = total

    return total, contrib

def score_frame_single_row(df_row, config, all_results):
    """단일 행에 대해 전체 종목 데이터를 참조하여 점수를 계산하는 함수"""
    if len(df_row) == 0:
        return pd.Series([0.0] * len(df_row), index=df_row.index), pd.DataFrame()
    
    # 1) 숫자형 피처 점수화 (전체 종목 데이터 참조)
    num_cfg = config.get("numeric", {})
    num_log = {}
    total_num = pd.Series(0.0, index=df_row.index)
    
    for feat, spec in num_cfg.items():
        if feat not in df_row.columns:
            print(f"Feature {feat} not found in DataFrame")
            continue
            
        try:
            # 현재 종목의 값
            current_value = df_row[feat].iloc[0]
            
            # 전체 종목에서 해당 피처의 값들을 수집
            all_values = []
            for result in all_results:
                if feat in result:
                    all_values.append(result[feat])
            
            # 전체 종목 데이터로 정규화
            if len(all_values) > 1:
                all_series = pd.Series(all_values)
                normalized_value = normalize_single_value(
                    current_value, 
                    all_series,
                    method=spec.get("norm", "rank"),
                    winsor=spec.get("winsor", [0.02, 0.98]),
                    clip=spec.get("clip"),
                    higher_is_better=spec.get("higher_is_better", True)
                )
            else:
                normalized_value = 50.0  # 기본값
            
            w = float(spec.get("weight", 0))
            total_num += normalized_value * w
            num_log[f"[N] {feat}"] = normalized_value * w
            print(f"Normalized {feat}: {normalized_value:.2f}, weight: {w}, contribution: {normalized_value * w:.2f}")
        except Exception as e:
            print(f"Error processing numeric feature {feat}: {e}")
            continue

    # 2) 룰 점수화
    rule_points, rule_log = apply_rules(df_row, config.get("rules", []))

    # 3) 합산
    total = total_num + rule_points
    print(f"Total score before filter: {total.values}")

    # 4) 필터 상한 적용 (디버깅 추가)
    cap_cfg = config.get("filters", {})
    if cap_cfg and cap_cfg.get("exprs"):
        pass_mask = np.ones(len(df_row), dtype=bool)
        print(f"Filter expressions: {cap_cfg['exprs']}")
        
        for expr in cap_cfg["exprs"]:
            try:
                result = eval_boolean_expr(df_row, expr)
                print(f"Filter '{expr}' result: {result.values}")
                pass_mask &= result.values
            except Exception as e:
                print(f"Filter '{expr}' error: {e}")
                pass_mask &= False
        
        print(f"Final pass_mask: {pass_mask}")
        cap_value = cap_cfg.get("cap_if_fail", 40.0)
        total = np.where(pass_mask, total, np.minimum(total, cap_value))
        total = pd.Series(total, index=df_row.index)
        print(f"Total score after filter: {total.values}")

    # 5) 기여도 로그 생성 (인덱스 명시적 지정)
    if num_log:
        num_df = pd.DataFrame([num_log], index=df_row.index)
    else:
        num_df = pd.DataFrame(index=df_row.index)
    
    if not rule_log.empty:
        contrib = pd.concat([num_df, rule_log], axis=1)
    else:
        contrib = num_df
    
    contrib["TOTAL"] = total

    return total, contrib

def normalize_single_value(value, all_values, method="rank", winsor=None, clip=None, higher_is_better=True):
    """단일 값을 전체 값들과 비교하여 정규화하는 함수"""
    if len(all_values) == 0:
        return 50.0
    
    # 극단값 완화
    if winsor:
        lo, hi = all_values.quantile(winsor[0]), all_values.quantile(winsor[1])
        all_values = all_values.clip(lo, hi)
        value = np.clip(value, lo, hi)
    
    # 정규화 방법
    if method == "minmax":
        rng = all_values.max() - all_values.min()
        if rng != 0:
            out = (value - all_values.min()) / rng
        else:
            out = 0.5
    elif method == "zscore":
        std = all_values.std(ddof=0)
        if std != 0:
            out = (value - all_values.mean()) / std
            # z-score를 0~100으로 매핑 (-3~+3 범위)
            out = (np.clip(out, -3, 3) + 3) / 6
        else:
            out = 0.5
    elif method == "rank":
        # 현재 값이 전체에서 몇 번째인지 계산
        rank = (all_values < value).sum() + 1
        out = rank / len(all_values)
    else:
        out = 0.5
    
    # 클리핑
    if clip:
        out = np.clip(out, clip[0], clip[1])
    
    # 0~100 스케일로 변환
    out = out * 100
    
    # 방향성 조정
    if not higher_is_better:
        out = 100 - out
    
    return out



# ---- 고급 스코어링 설정 (지시문 기반 개선)
ADVANCED_SCORE_CONFIG = {
    "numeric": {
        # 상승률 지표 (높을수록 좋음)
        "return_20d": {"weight": 0.18, "norm": "rank", "winsor": [0.02, 0.98], "higher_is_better": True},
        "return_60d": {"weight": 0.12, "norm": "rank", "winsor": [0.02, 0.98], "higher_is_better": True},
        
        # 상대 모멘텀 지표 (핵심 가중치)
        "rel_mom_20": {"weight": 0.20, "norm": "rank", "higher_is_better": True},
        "rel_mom_60": {"weight": 0.15, "norm": "rank", "higher_is_better": True},
        "rel_mom_120": {"weight": 0.15, "norm": "rank", "higher_is_better": True},
        
        # 위험조정수익률 지표 (퀄리티 레이어)
        "sharpe_ratio_60": {"weight": 0.10, "norm": "rank", "higher_is_better": True},
        "sortino_ratio_60": {"weight": 0.08, "norm": "rank", "higher_is_better": True},
        "calmar_ratio_120": {"weight": 0.08, "norm": "rank", "higher_is_better": True},
        
        # 변동성 지표 (낮을수록 좋음)
        "volatility_20": {"weight": 0.07, "norm": "rank", "higher_is_better": False},
        "volatility_60": {"weight": 0.05, "norm": "rank", "higher_is_better": False},
        
        # 볼린저 밴드 (극단 회귀 모드)
        "bb_position": {"weight": 0.05, "norm": "minmax", "mode": "revert", "higher_is_better": False},
        "bb_width": {"weight": 0.03, "norm": "rank", "higher_is_better": False},
        
        # 거래량 지표
        "volume_ratio": {"weight": 0.07, "norm": "rank", "higher_is_better": True},
        
        # 52주 고점/저점 (돌파 추종 모드)
        "high_52w_ratio": {"weight": 0.15, "norm": "zscore", "clip": [-2.5, 2.5], "higher_is_better": True},
        "low_52w_ratio": {"weight": 0.05, "norm": "rank", "higher_is_better": False},
        
        # 기술적 지표
        "rsi": {"weight": 0.05, "norm": "minmax", "higher_is_better": False},  # 50 근처가 좋음
    },
    
    "rules": [
        # 추세 필터 (핵심)
        {
            "name": "TrendFilter",
            "type": "boolean",
            "expr": "sma20 > sma60 * 1.002 and close > sma20",
            "score": 8.0
        },
        
        # 이동평균 정렬
        {
            "name": "MAAlignment",
            "type": "boolean",
            "expr": "sma5 > sma20 and sma20 > sma60",
            "score": 6.0
        },
        
        # RSI 건강한 구간
        {
            "name": "RSI_Healthy",
            "type": "boolean",
            "expr": "rsi >= 50 and rsi <= 70",
            "score": 4.0
        },
        
        # RSI 과매도 (반등 기대)
        {
            "name": "RSI_Oversold",
            "type": "boolean",
            "expr": "rsi < 30",
            "score": 5.0
        },
        
        # MACD 상향 전환
        {
            "name": "MACD_TurnUp",
            "type": "boolean",
            "expr": "macd_histogram > 0 and macd_histogram.shift(1) <= 0",
            "score": 6.0
        },
        
        # 거래량 급증
        {
            "name": "Volume_Surge",
            "type": "boolean",
            "expr": "volume_ratio > 1.5",
            "score": 4.0
        },
        
        # 골든크로스 이벤트 (감쇠 적용)
        {
            "name": "Golden_20_60",
            "type": "event_decay",
            "event": "CROSSOVER(sma20, sma60)",
            "base": 12.0,
            "tau": 5.0
        },
        
        # 데드크로스 이벤트 (감쇠 적용)
        {
            "name": "Dead_20_60",
            "type": "event_decay",
            "event": "CROSSUNDER(sma20, sma60)",
            "base": -14.0,
            "tau": 5.0
        },
        
        # 볼린저 밴드 하단 반등
        {
            "name": "BB_Bounce",
            "type": "boolean",
            "expr": "bb_position < 0.2 and close > close.shift(1)",
            "score": 5.0
        },
        
        # 52주 고점 근접
        {
            "name": "52W_High_Near",
            "type": "boolean",
            "expr": "high_52w_ratio > 95 and high_52w_ratio < 99",
            "score": 3.0
        }
    ],
    
    "filters": {
        "cap_if_fail": 40.0,
        "exprs": [
            "close > sma20",
            "sma20 > sma60"
        ]
    },
    
    "thresholds": {
        "buy": 70,
        "hold": 55,
        "sell": 50
    }
}

# 단순화된 스코어링 설정 (디버깅용)
SIMPLE_SCORE_CONFIG = {
    "numeric": {
        # 기본 지표들만 사용
        "return_20d": {"weight": 0.3, "norm": "rank", "higher_is_better": True},
        "rel_mom_20": {"weight": 0.3, "norm": "rank", "higher_is_better": True},
        "rsi": {"weight": 0.2, "norm": "minmax", "higher_is_better": False},
        "volume_ratio": {"weight": 0.2, "norm": "rank", "higher_is_better": True},
    },
    "rules": [
        {
            "name": "TrendFilter",
            "type": "boolean",
            "expr": "sma20 > sma60",
            "score": 10.0
        }
    ],
    "filters": {
        "cap_if_fail": 50.0,
        "exprs": []  # 빈 리스트로 설정
    }
}

# ---- 스코어링 결과 분석 헬퍼
def analyze_score_contribution(contrib_df, top_n=5):
    """스코어 기여도 분석"""
    if contrib_df.empty:
        return pd.DataFrame()
    
    # 평균 기여도 계산
    avg_contrib = contrib_df.mean().sort_values(ascending=False)
    
    # 상위 기여도만 선택
    top_contrib = avg_contrib.head(top_n)
    
    return pd.DataFrame({
        'Feature': top_contrib.index,
        'Avg_Contribution': top_contrib.values,
        'Percentage': (top_contrib.values / top_contrib.sum() * 100).round(2)
    })

# 주식 종목 코드와 이름 매핑 (KODEX 200 제외, 실제 파일 기반으로 업데이트)
CODE_TO_NAME = {
    "000270": "Kia",
    "000660": "SK Hynix",
    "000810": "삼성화재",
    "005380": "Hyundai Motor",
    "005490": "POSCO",
    "005930": "Samsung Electronics",
    "005935": "Samsung Electronics (우선주)",
    "009540": "현대중공업",
    "011200": "HMM",
    "012330": "현대모비스",
    "012450": "한화에어로스페이스",
    "015760": "한국전력",
    "028260": "삼성물산",
    "032830": "삼성생명",
    "034020": "Doosan Enerbility",
    "035420": "NAVER",
    "035720": "카카오",
    "042660": "대우건설",
    "051910": "LG Chem",
    "055550": "신한지주",
    "064350": "현대로템",
    "068270": "Celltrion",
    "086790": "하나금융지주",
    "105560": "KB Financial",
    "138040": "메리츠금융지주",
    "207940": "Samsung Biologics",
    "316140": "우리금융지주",
    "329180": "HD Hyundai Construction Equipment",
    "373220": "LG Energy Solution",
    "402340": "SK스퀘어"
}

DATA_FOLDER = "/home/hanmil/backtest_app2"
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# KODEX 200 데이터에서 거래일 추출
def get_trading_dates():
    try:
        df_kodex = pd.read_csv(os.path.join(DATA_FOLDER, "069500_daily_price.csv"))
        date_col = find_column(df_kodex, ['거래일자', 'date', 'Date', '날짜'])
        if date_col:
            # YYYYMMDD 형식을 올바르게 파싱
            df_kodex[date_col] = pd.to_datetime(df_kodex[date_col], format='%Y%m%d')
            trading_dates = df_kodex[date_col].dt.date.unique()
            trading_dates = sorted(trading_dates)
            return trading_dates
        else:
            st.error("날짜 컬럼을 찾을 수 없습니다.")
            return []
    except Exception as e:
        st.error(f"KODEX 200 데이터 로드 중 오류: {e}")
        return []

# 거래일 목록 가져오기
trading_dates = get_trading_dates()

# 디버깅 정보 추가
st.write(f"**전체 거래일 수**: {len(trading_dates) if trading_dates else 0}")

if trading_dates:
    # 거래일 범위 계산
    min_date = min(trading_dates)
    max_date = max(trading_dates)
    
    # 2023년 7월 3일부터 2025년 6월 30일까지의 거래일 필터링
    start_limit = dt.date(2023, 7, 3)
    end_limit = dt.date(2025, 6, 30)
    filtered_trading_dates = [d for d in trading_dates if start_limit <= d <= end_limit]
    
    st.write(f"**필터링된 거래일 수**: {len(filtered_trading_dates)}")
    st.write(f"**전체 데이터 범위**: {min_date} ~ {max_date}")
    st.write(f"**필터링 범위**: {start_limit} ~ {end_limit}")
    
    if filtered_trading_dates:
        # 연도별, 월별, 일별로 거래일 그룹화
        years = sorted(list(set(d.year for d in filtered_trading_dates)))
        years_str = [str(year) for year in years]
        
        # 시작일과 종료일 선택
        st.subheader("기간 선택")
        st.write("거래일만 선택 가능합니다 (2023년 7월 3일 ~ 2025년 6월 30일)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**시작일**")
            start_year = st.selectbox("연도", years_str, index=0, key="start_year")
            start_year_dates = [d for d in filtered_trading_dates if d.year == int(start_year)]
            start_months = sorted(list(set(d.month for d in start_year_dates)))
            start_months_str = [f"{month:02d}월" for month in start_months]
            start_month = st.selectbox("월", start_months_str, key="start_month")
            start_month_num = int(start_month.replace("월", ""))
            start_month_dates = [d for d in start_year_dates if d.month == start_month_num]
            start_days = sorted(list(set(d.day for d in start_month_dates)))
            start_days_str = [f"{day:02d}일" for day in start_days]
            start_day = st.selectbox("일", start_days_str, key="start_day")
            start_day_num = int(start_day.replace("일", ""))
            start_date = dt.date(int(start_year), start_month_num, start_day_num)
            
        with col2:
            st.write("**종료일**")
            end_year = st.selectbox("연도", years_str, index=len(years_str)-1, key="end_year")
            end_year_dates = [d for d in filtered_trading_dates if d.year == int(end_year)]
            end_months = sorted(list(set(d.month for d in end_year_dates)))
            end_months_str = [f"{month:02d}월" for month in end_months]
            end_month = st.selectbox("월", end_months_str, index=len(end_months_str)-1, key="end_month")
            end_month_num = int(end_month.replace("월", ""))
            end_month_dates = [d for d in end_year_dates if d.month == end_month_num]
            end_days = sorted(list(set(d.day for d in end_month_dates)))
            end_days_str = [f"{day:02d}일" for day in end_days]
            end_day = st.selectbox("일", end_days_str, index=len(end_days_str)-1, key="end_day")
            end_day_num = int(end_day.replace("일", ""))
            end_date = dt.date(int(end_year), end_month_num, end_day_num)
        
        # 날짜 유효성 검사
        if start_date > end_date:
            st.error("시작일이 종료일보다 늦습니다. 올바른 기간을 선택해주세요.")
        else:
            st.success(f"선택된 기간: {start_date} ~ {end_date}")

            # 거래일 정보 표시
            with st.expander("거래일 정보"):
                st.write(f"**전체 데이터 범위**: {min_date} ~ {max_date}")
                st.write(f"**선택 가능 범위**: {start_limit} ~ {end_limit}")
                st.write(f"**선택 가능한 거래일 수**: {len(filtered_trading_dates)}일")
                st.write(f"**선택된 기간**: {start_date} ~ {end_date}")
                
                # 선택된 기간의 거래일 수 계산
                selected_trading_dates = [d for d in filtered_trading_dates if start_date <= d <= end_date]
                st.write(f"**선택된 기간 거래일 수**: {len(selected_trading_dates)}일")
    else:
        st.error("필터링된 거래일이 없습니다. 데이터 범위를 확인해주세요.")
else:
    st.error("거래일 데이터를 가져올 수 없습니다.")

# ==============================
# 주식 종목 선택 (Opt-out 방식)
# ==============================
st.subheader("주식 종목 선택")

file_paths = sorted(glob(os.path.join(DATA_FOLDER, "*_daily_price.csv")))
stock_codes = [os.path.basename(p).split("_")[0] for p in file_paths if os.path.basename(p).split("_")[0] != "069500"]  # KODEX 200 제외
stock_names = [f"{CODE_TO_NAME.get(code, code)} ({code})" for code in stock_codes]

# 기본값으로 모든 종목 선택
default_selected = stock_names.copy()

# 제외할 종목 선택 (Opt-out)
excluded_stocks = st.multiselect(
    "제외할 종목 선택 (기본값: 모든 종목 선택)", 
    options=stock_names,
    help="선택한 종목들은 분석에서 제외됩니다. 아무것도 선택하지 않으면 모든 종목이 분석 대상입니다."
)

# 최종 선택된 종목 계산
selected_stocks = [name for name in default_selected if name not in excluded_stocks]
selected_codes = [name.split("(")[-1][:-1] for name in selected_stocks]

st.write(f"**분석 대상 종목 수**: {len(selected_codes)}개")
if selected_codes:
    st.write(f"**분석 대상**: {', '.join([CODE_TO_NAME.get(code, code) for code in selected_codes[:5]])}{'...' if len(selected_codes) > 5 else ''}")

# ==============================
# 사용 가능한 Feature 목록
# ==============================
st.subheader("사용 가능한 Feature 목록")

# Feature 카테고리별로 표시
feature_categories = {
    "기본 가격 데이터": ["close", "open", "high", "low", "volume"],
    "이동평균선": ["sma5", "sma10", "sma20", "sma60", "sma120"],
    "지수이동평균선": ["ema12", "ema26"],
    "기술적 지표": ["rsi", "macd", "macd_signal", "macd_histogram"],
    "볼린저 밴드": ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"],
    "거래량 지표": ["volume_sma5", "volume_sma20", "volume_ratio"],
    "변동성 지표": ["atr", "volatility_20", "volatility_60", "volatility_annualized_20", "volatility_annualized_60", "parkinson_volatility", "garman_klass_volatility", "true_range_volatility"],
    "상승률 지표": ["return_2d", "return_20d", "return_60d"],
    "상대 모멘텀 지표": ["rel_mom_20", "rel_mom_60", "rel_mom_120"],
    "52주 고점/저점 지표": ["high_52w_ratio", "low_52w_ratio"],
    "위험조정수익률 지표": ["sharpe_ratio_20", "sharpe_ratio_60", "sharpe_ratio_120", "sortino_ratio_20", "sortino_ratio_60", "sortino_ratio_120", "calmar_ratio_20", "calmar_ratio_60", "calmar_ratio_120"]
}

for category, features in feature_categories.items():
    with st.expander(f"{category} ({len(features)}개)"):
        # 3열로 표시
        cols = st.columns(3)
        for i, feature in enumerate(features):
            col_idx = i % 3
            with cols[col_idx]:
                st.write(f"• **{feature}**")
        
        # 설명 추가
        if category == "기본 가격 데이터":
            st.write("**설명**: 주식의 기본 가격 정보 (종가, 시가, 고가, 저가, 거래량)")
        elif category == "이동평균선":
            st.write("**설명**: 단순이동평균선 (Simple Moving Average)")
        elif category == "지수이동평균선":
            st.write("**설명**: 지수이동평균선 (Exponential Moving Average)")
        elif category == "기술적 지표":
            st.write("**설명**: RSI (상대강도지수), MACD (이동평균수렴확산지수)")
        elif category == "볼린저 밴드":
            st.write("**설명**: 볼린저 밴드 (Bollinger Bands) - 변동성 기반 지표")
        elif category == "거래량 지표":
            st.write("**설명**: 거래량 기반 지표 (거래량 이동평균, 거래량 비율)")
        elif category == "변동성 지표":
            st.write("**설명**: 다양한 변동성 계산 방법 (ATR, Parkinson, Garman-Klass 등)")
        elif category == "상승률 지표":
            st.write("**설명**: 과거 대비 가격 상승률 (2일, 20일, 60일)")
        elif category == "상대 모멘텀 지표":
            st.write("**설명**: KODEX 200 대비 상대적 모멘텀")
        elif category == "52주 고점/저점 지표":
            st.write("**설명**: 52주 고점/저점 대비 현재가 비율")
        elif category == "위험조정수익률 지표":
            st.write("**설명**: 위험 대비 초과수익률을 측정하는 지표 (Sharpe, Sortino, Calmar)")

# Feature 사용 예시
with st.expander("Feature 사용 예시"):
    st.write("**조건 설정 예시**:")
    st.write("• `rsi < 30` : RSI가 30 미만일 때 (과매도)")
    st.write("• `rsi > 70` : RSI가 70 초과일 때 (과매수)")
    st.write("• `close > sma20` : 종가가 20일 이동평균선 위에 있을 때")
    st.write("• `volume_ratio > 2` : 거래량이 20일 평균의 2배 이상일 때")
    st.write("• `volatility_20 > 3` : 20일 변동성이 3% 이상일 때")
    st.write("• `high_52w_ratio > 95` : 52주 고점의 95% 이상일 때")
    st.write("• `rel_mom_20 > 5` : KODEX 200 대비 20일 상대 모멘텀이 5% 이상일 때")

# ==============================
# 점수 계산 설정 (사용자 정의)
# ==============================
st.subheader("점수 계산 설정")

# 사용 가능한 지표 목록
available_indicators = {
    "상승률 지표": {
        "return_2d": "2일 상승률",
        "return_20d": "20일 상승률", 
        "return_60d": "60일 상승률"
    },
    "상대 모멘텀 지표": {
        "rel_mom_20": "20일 상대 모멘텀",
        "rel_mom_60": "60일 상대 모멘텀",
        "rel_mom_120": "120일 상대 모멘텀"
    },
    "기술적 지표": {
        "rsi": "RSI",
        "macd": "MACD",
        "macd_histogram": "MACD 히스토그램"
    },
    "이동평균선": {
        "sma5": "5일 이동평균",
        "sma20": "20일 이동평균",
        "sma60": "60일 이동평균"
    },
    "볼린저 밴드": {
        "bb_position": "볼린저 밴드 위치",
        "bb_width": "볼린저 밴드 폭"
    },
    "거래량 지표": {
        "volume_ratio": "거래량 비율",
        "volume_sma5": "거래량 5일 평균",
        "volume_sma20": "거래량 20일 평균"
    },
    "변동성 지표": {
        "volatility_20": "20일 변동성",
        "volatility_60": "60일 변동성",
        "atr": "ATR"
    },
    "52주 고점/저점": {
        "high_52w_ratio": "52주 고점 비율",
        "low_52w_ratio": "52주 저점 비율"
    },
    "위험조정수익률": {
        "sharpe_ratio_20": "20일 샤프 비율",
        "sharpe_ratio_60": "60일 샤프 비율",
        "sortino_ratio_60": "60일 소르티노 비율",
        "calmar_ratio_120": "120일 칼마 비율"
    }
}

# 기본 설정
default_indicators = {
    "return_20d": {"weight": 0.25, "norm": "rank", "higher_is_better": True},
    "rel_mom_20": {"weight": 0.25, "norm": "rank", "higher_is_better": True},
    "rsi": {"weight": 0.20, "norm": "minmax", "higher_is_better": False},
    "volume_ratio": {"weight": 0.15, "norm": "rank", "higher_is_better": True},
    "sharpe_ratio_60": {"weight": 0.15, "norm": "rank", "higher_is_better": True}
}

# 사용자 정의 설정
st.write("**지표 선택 및 가중치 설정**")
st.write("가중치 총합이 1.0이 되도록 설정해주세요.")

# 탭으로 카테고리별 구분
tab_names = list(available_indicators.keys())
tabs = st.tabs(tab_names)

# session_state에서 selected_indicators 초기화
if 'selected_indicators' not in st.session_state:
    st.session_state.selected_indicators = {}

selected_indicators = st.session_state.selected_indicators
total_weight = 0.0

for i, (category, indicators) in enumerate(available_indicators.items()):
    with tabs[i]:
        st.write(f"**{category}**")
        
        for indicator_code, indicator_name in indicators.items():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                # 지표 선택 체크박스
                is_selected = st.checkbox(
                    indicator_name, 
                    value=indicator_code in default_indicators,
                    key=f"select_{indicator_code}"
                )
            
            if is_selected:
                with col2:
                    # 가중치 입력
                    weight = st.number_input(
                        "가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_indicators.get(indicator_code, {}).get("weight", 0.1),
                        step=0.05,
                        format="%.2f",
                        key=f"weight_{indicator_code}"
                    )
                
                with col3:
                    # 정규화 방법 선택
                    norm_method = st.selectbox(
                        "정규화",
                        options=["rank", "minmax", "zscore"],
                        index=0 if default_indicators.get(indicator_code, {}).get("norm", "rank") == "rank" else 
                              1 if default_indicators.get(indicator_code, {}).get("norm", "rank") == "minmax" else 2,
                        key=f"norm_{indicator_code}"
                    )
                
                with col4:
                    # 방향성 선택
                    higher_better = st.checkbox(
                        "높을수록 좋음",
                        value=default_indicators.get(indicator_code, {}).get("higher_is_better", True),
                        key=f"direction_{indicator_code}"
                    )
                
                selected_indicators[indicator_code] = {
                    "weight": weight,
                    "norm": norm_method,
                    "higher_is_better": higher_better
                }
                total_weight += weight
            else:
                # 지표 선택이 취소된 경우 selected_indicators에서 제거
                if indicator_code in selected_indicators:
                    del selected_indicators[indicator_code]
            
            # session_state 업데이트 (선택/취소 모두 반영)
            st.session_state.selected_indicators = selected_indicators



# ==============================
# 사용자 정의 지표 추가
# ==============================
st.subheader("사용자 정의 지표 추가")

# 사용 가능한 컬럼 목록 표시
st.write("**사용 가능한 컬럼:**")
basic_columns = [
    'close', 'open', 'high', 'low', 'volume',
    'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
    'ema12', 'ema26', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'volume_ratio', 'atr', 'volatility_20', 'volatility_60',
    'rel_mom_20', 'rel_mom_60', 'rel_mom_120',
    'high_52w_ratio', 'low_52w_ratio',
    'sharpe_ratio_20', 'sharpe_ratio_60', 'sortino_ratio_60', 'calmar_ratio_120',
    'return_2d', 'return_20d', 'return_60d'
]

additional_columns = st.session_state.get('all_columns', [])
all_columns = list(set(basic_columns + additional_columns))

st.code(", ".join(sorted(all_columns)))
st.write("💡 **팁**: 위 컬럼들을 사용하여 조건식을 작성하세요. 예: `volume / close`, `(close - sma20) / sma20 * 100`")

# 조건식 검증 함수
def validate_expression(expr, available_columns=None):
    """조건식이 유효한지 검증"""
    if not expr:
        return False, "조건식을 입력해주세요."
    
    try:
        # 기본 컬럼들 (항상 사용 가능)
        basic_columns = [
            'close', 'open', 'high', 'low', 'volume',
            'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
            'ema12', 'ema26', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'volume_ratio', 'atr', 'volatility_20', 'volatility_60',
            'rel_mom_20', 'rel_mom_60', 'rel_mom_120',
            'high_52w_ratio', 'low_52w_ratio',
            'sharpe_ratio_20', 'sharpe_ratio_60', 'sortino_ratio_60', 'calmar_ratio_120',
            'return_2d', 'return_20d', 'return_60d'
        ]
        
        # 사용 가능한 컬럼 목록 (기본 + 추가)
        if available_columns:
            all_columns = list(set(basic_columns + available_columns))
        else:
            all_columns = basic_columns
        
        # 테스트용 더미 데이터 생성
        test_data = {}
        for col in all_columns:
            test_data[col] = [1.0, 2.0, 3.0]  # 테스트용 값
        
        test_df = pd.DataFrame(test_data)
        
        # 조건식 평가 시도
        result = pd.eval(expr, local_dict=test_df.to_dict(orient="series"))
        
        if len(result) == 3:  # 예상된 길이
            return True, "✅ 조건식이 유효합니다."
        else:
            return False, "❌ 조건식 결과가 예상과 다릅니다."
            
    except Exception as e:
        return False, f"❌ 조건식 오류: {str(e)}"

# 사용자 정의 지표 입력
custom_indicator_name = st.text_input("지표 이름", placeholder="예: 가격대비거래량비율", key="custom_indicator_name_1")
custom_indicator_expr = st.text_input("조건식", placeholder="예: volume / close", key="custom_indicator_expr_1")

# 조건식 검증 (항상 실행)
if custom_indicator_expr:
    available_columns = st.session_state.get('all_columns', [])
    is_valid, validation_msg = validate_expression(custom_indicator_expr, available_columns)
    
    # 검증 결과 표시
    if is_valid:
        st.success(validation_msg)
    else:
        st.error(validation_msg)
    
    if is_valid:
        # 사용자 정의 지표 설정
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            custom_weight = st.number_input(
                "가중치",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="custom_weight"
            )
        
        with col2:
            custom_norm = st.selectbox(
                "정규화 방법",
                options=["rank", "minmax", "zscore"],
                key="custom_norm"
            )
        
        with col3:
            custom_direction = st.checkbox(
                "높을수록 좋음",
                value=True,
                key="custom_direction"
            )
        
        with col4:
            if st.button("지표 추가", key="add_custom"):
                # 사용자 정의 지표를 selected_indicators에 추가
                custom_key = f"custom_{custom_indicator_name}"
                selected_indicators[custom_key] = {
                    "weight": custom_weight,
                    "norm": custom_norm,
                    "higher_is_better": custom_direction,
                    "expression": custom_indicator_expr,  # 실제 계산에 사용할 식
                    "name": custom_indicator_name  # 표시용 이름
                }
                # session_state 업데이트
                st.session_state.selected_indicators = selected_indicators
                st.success(f"'{custom_indicator_name}' 지표가 추가되었습니다!")
                st.rerun()

# 사용자 정의 지표 목록 표시
custom_indicators = {k: v for k, v in selected_indicators.items() if k.startswith('custom_')}
if custom_indicators:
    st.write("**추가된 사용자 정의 지표:**")
    for key, config in custom_indicators.items():
        st.write(f"- {config['name']}: {config['expression']} (가중치: {config['weight']:.2f})")

# 전체 가중치 총합 재계산 (기본 지표 + 사용자 정의 지표)
total_weight = sum(config.get("weight", 0) for config in selected_indicators.values())

# 가중치 총합 표시 및 경고
st.write("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("선택된 지표 수", len(selected_indicators))
with col2:
    st.metric("가중치 총합", f"{total_weight:.2f}")
    
if abs(total_weight - 1.0) > 0.01:
    st.warning(f"⚠️ 가중치 총합이 1.0이 아닙니다 ({total_weight:.2f}). 정확한 점수 계산을 위해 1.0으로 맞춰주세요.")
else:
    st.success("✅ 가중치 총합이 1.0입니다.")

# 자동 가중치 조정 옵션
if abs(total_weight - 1.0) > 0.01 and total_weight > 0:
    if st.button("가중치 자동 조정 (총합을 1.0으로)"):
        for indicator in selected_indicators:
            selected_indicators[indicator]["weight"] /= total_weight
        # session_state 업데이트
        st.session_state.selected_indicators = selected_indicators
        st.success("가중치가 자동으로 조정되었습니다!")
        st.rerun()

def calculate_custom_indicators(df, custom_indicators):
    """사용자 정의 지표를 계산하는 함수"""
    result_df = df.copy()
    
    for indicator_key, config in custom_indicators.items():
        if 'expression' in config:
            try:
                # 사용자 정의 식 계산
                expr = config['expression']
                calculated_values = pd.eval(expr, local_dict=result_df.to_dict(orient="series"))
                result_df[indicator_key] = calculated_values
            except Exception as e:
                print(f"사용자 정의 지표 '{config['name']}' 계산 오류: {e}")
                result_df[indicator_key] = np.nan
    
    return result_df



# ==============================
# 필터 및 트리거 설정
# ==============================
st.subheader("필터 및 트리거 설정")

# 기본 스코어링 설정 사용
scoring_config = ADVANCED_SCORE_CONFIG

# 필터 설정
st.write("**📊 필터 설정 (AND 조건 - 모든 조건을 만족해야 함)**")
st.write("미통과 시 총점 상한(cap_if_fail)이 적용됩니다.")

filter_exprs = []
col1, col2 = st.columns(2)

with col1:
    # 추세 필터
    trend_filter = st.checkbox("추세 필터", value=True, help="sma20 > sma60 and close > sma20")
    if trend_filter:
        filter_exprs.append("sma20 > sma60 * 1.002 and close > sma20")
    
    # 이동평균 정렬 필터
    ma_alignment_filter = st.checkbox("이동평균 정렬 필터", value=False, help="sma5 > sma20 and sma20 > sma60")
    if ma_alignment_filter:
        filter_exprs.append("sma5 > sma20 and sma20 > sma60")

with col2:
    # 거래량 필터
    volume_filter = st.checkbox("거래량 필터", value=False, help="volume_ratio > 0.5")
    if volume_filter:
        filter_exprs.append("volume_ratio > 0.5")
    
    # RSI 필터
    rsi_filter = st.checkbox("RSI 필터", value=False, help="rsi >= 30 and rsi <= 70")
    if rsi_filter:
        filter_exprs.append("rsi >= 30 and rsi <= 70")

# 필터 실패 시 상한값 설정
cap_if_fail = st.slider("필터 실패 시 총점 상한", min_value=0.0, max_value=100.0, value=40.0, step=5.0, help="필터 조건을 만족하지 않을 때 적용될 최대 점수")

# 트리거 설정
st.write("**🚀 트리거 설정 (OR 조건 - 하나라도 만족하면 가산점)**")

# 트리거 목록
triggers = []

# 골든크로스 트리거
golden_cross = st.checkbox("골든크로스 트리거", value=True, help="sma20이 sma60을 상향돌파할 때")
if golden_cross:
    golden_cross_score = st.number_input("골든크로스 가산점", min_value=0.0, max_value=20.0, value=12.0, step=1.0)
    golden_cross_tau = st.number_input("골든크로스 감쇠기간(tau)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    triggers.append({
        "name": "Golden_20_60",
        "type": "event_decay",
        "event": "CROSSOVER(sma20, sma60)",
        "base": golden_cross_score,
        "tau": golden_cross_tau
    })

# MACD 양전환 트리거
macd_turnup = st.checkbox("MACD 양전환 트리거", value=True, help="MACD 히스토그램이 음수에서 양수로 전환할 때")
if macd_turnup:
    macd_score = st.number_input("MACD 양전환 가산점", min_value=0.0, max_value=20.0, value=6.0, step=1.0)
    triggers.append({
        "name": "MACD_TurnUp",
        "type": "boolean",
        "expr": "macd_histogram > 0 and macd_histogram.shift(1) <= 0",
        "score": macd_score
    })

# 볼린저 밴드 돌파 트리거
bb_breakout = st.checkbox("볼린저 밴드 상단 돌파 트리거", value=False, help="close가 bb_upper를 상향돌파할 때")
if bb_breakout:
    bb_score = st.number_input("볼린저 밴드 돌파 가산점", min_value=0.0, max_value=20.0, value=5.0, step=1.0)
    triggers.append({
        "name": "BB_Breakout",
        "type": "boolean",
        "expr": "close > bb_upper and close.shift(1) <= bb_upper.shift(1)",
        "score": bb_score
    })

# 거래량 급증 트리거
volume_surge = st.checkbox("거래량 급증 트리거", value=True, help="거래량이 20일 평균의 1.5배 이상일 때")
if volume_surge:
    volume_surge_threshold = st.number_input("거래량 급증 기준", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    volume_surge_score = st.number_input("거래량 급증 가산점", min_value=0.0, max_value=20.0, value=4.0, step=1.0)
    triggers.append({
        "name": "Volume_Surge",
        "type": "boolean",
        "expr": f"volume_ratio > {volume_surge_threshold}",
        "score": volume_surge_score
    })

# RSI 과매도 반등 트리거
rsi_bounce = st.checkbox("RSI 과매도 반등 트리거", value=False, help="RSI가 30 이하에서 반등할 때")
if rsi_bounce:
    rsi_bounce_score = st.number_input("RSI 반등 가산점", min_value=0.0, max_value=20.0, value=5.0, step=1.0)
    triggers.append({
        "name": "RSI_Bounce",
        "type": "boolean",
        "expr": "rsi < 30 and rsi > rsi.shift(1)",
        "score": rsi_bounce_score
    })

# 설정된 필터와 트리거를 스코어링 설정에 적용
scoring_config["filters"]["exprs"] = filter_exprs
scoring_config["filters"]["cap_if_fail"] = cap_if_fail
scoring_config["rules"].extend(triggers)

# 설정 요약 표시
st.write("**📋 설정 요약**")
st.write(f"**필터 조건**: {len(filter_exprs)}개")
for i, expr in enumerate(filter_exprs, 1):
    st.write(f"{i}. {expr}")
st.write(f"**필터 실패 시 상한**: {cap_if_fail:.1f}점")
st.write(f"**트리거 조건**: {len(triggers)}개")
for trigger in triggers:
    st.write(f"- {trigger['name']}: {trigger.get('score', f'base={trigger.get('base', 0)}')}점")

# ==============================
# 분석 실행
# ==============================
st.subheader("분석 실행")

# session_state 초기화
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False

if st.button("분석 시작") or st.session_state.analysis_completed:
    if not selected_codes:
        st.warning("분석할 종목을 선택해주세요.")
    else:
        # 이미 분석이 완료되었고 결과가 있으면 재분석하지 않음
        if st.session_state.analysis_completed and st.session_state.analysis_results is not None:
            st.write("**이전 분석 결과를 사용합니다.**")
            all_results = st.session_state.analysis_results
        else:
            st.write("**선택된 종목**:", selected_codes)
        
        # 선택된 기간의 거래일만 필터링
        selected_trading_dates = [d for d in filtered_trading_dates if start_date <= d <= end_date]
        st.write(f"**분석 기간**: {len(selected_trading_dates)}일")
        
        # KODEX 200 데이터 로드
        try:
            df_kodex = pd.read_csv(os.path.join(DATA_FOLDER, "069500_daily_price.csv"))
        except Exception as e:
            st.error(f"KODEX 200 데이터 로드 실패: {e}")
            st.stop()
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 결과를 저장할 리스트
        all_results = []
        
        # 디버깅 정보 추가
        st.write(f"**분석 시작**: {len(selected_codes)}개 종목, {len(selected_trading_dates)}일")
        st.write(f"**선택된 종목**: {selected_codes[:5]}{'...' if len(selected_codes) > 5 else ''}")
        st.write(f"**분석 기간**: {selected_trading_dates[0]} ~ {selected_trading_dates[-1]}")
        
        for i, trading_date in enumerate(selected_trading_dates):
            status_text.text(f"분석 중... {i+1}/{len(selected_trading_dates)}일차 ({trading_date})")
            progress_bar.progress((i + 1) / len(selected_trading_dates))
            
            daily_results = []
            
            for code in selected_codes:
                try:
                    # 파일 존재 여부 확인
                    file_path = os.path.join(DATA_FOLDER, f"{code}_daily_price.csv")
                    if not os.path.exists(file_path):
                        print(f"파일이 존재하지 않음: {file_path}")
                        continue
                        
                    df = pd.read_csv(file_path)
                    print(f"파일 로드 성공: {code}, 데이터 크기: {df.shape}")
                    date_col = find_column(df, ['거래일자', 'date', 'Date', '날짜'])
                    close_col = find_column(df, ['종가', 'close', 'Close'])
                    high_col = find_column(df, ['고가', 'high', 'High'])
                    low_col = find_column(df, ['저가', 'low', 'Low'])
                    volume_col = find_column(df, ['거래량', 'volume', 'Volume'])
                    
                    # 컬럼 찾기 디버깅
                    print(f"컬럼 찾기 결과 - {code}: date={date_col}, close={close_col}, high={high_col}, low={low_col}, volume={volume_col}")
                    print(f"실제 컬럼 목록: {list(df.columns)}")
                    
                    # YYYYMMDD 형식으로 파싱
                    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
                    
                    # 해당 날짜까지의 데이터만 사용
                    df_filtered = df[df[date_col] <= pd.to_datetime(trading_date)].copy()
                    
                    print(f"데이터 필터링 - {code}: 원본 {len(df)}행 -> 필터링 {len(df_filtered)}행")
                    
                    if len(df_filtered) > 0:
                        # 상승률 계산
                        df_filtered = calculate_returns(df_filtered, date_col, close_col)
                        
                        # 이동평균선 계산
                        df_filtered = calculate_sma(df_filtered, close_col)
                        df_filtered = calculate_ema(df_filtered, close_col)
                        
                        # 기술적 지표 계산
                        df_filtered = calculate_rsi(df_filtered, close_col)
                        df_filtered = calculate_macd(df_filtered, close_col)
                        
                        # 거래량 지표 계산
                        if volume_col:
                            df_filtered = calculate_volume_indicators(df_filtered, volume_col, close_col)
                        
                        # 볼린저 밴드 계산
                        df_filtered = calculate_bollinger_bands(df_filtered, close_col)
                        
                        # ATR 계산
                        if high_col and low_col:
                            df_filtered = calculate_atr(df_filtered, high_col, low_col, close_col)
                        
                        # 변동성 지표 계산
                        df_filtered = calculate_volatility(df_filtered, close_col)
                        
                        # OHLC 기반 변동성 계산
                        if high_col and low_col:
                            open_col = find_column(df_filtered, ['시가', 'open', 'Open'])
                            if open_col:
                                df_filtered = calculate_parkinson_volatility(df_filtered, high_col, low_col)
                                df_filtered = calculate_garman_klass_volatility(df_filtered, open_col, high_col, low_col, close_col)
                                df_filtered = calculate_true_range_volatility(df_filtered, high_col, low_col, close_col)
                        
                        # 상대 모멘텀 계산 (KODEX 200 대비)
                        df_filtered = calculate_relative_momentum(df_filtered, df_kodex, date_col, close_col)
                        
                        # 52주 고점/저점 계산
                        if high_col and low_col:
                            df_filtered = calculate_52week_high_low(df_filtered, date_col, close_col, high_col, low_col)
                        
                        # 위험조정수익률 지표 계산
                        df_filtered = calculate_sharpe_ratio(df_filtered, close_col)
                        df_filtered = calculate_sortino_ratio(df_filtered, close_col)
                        df_filtered = calculate_calmar_ratio(df_filtered, close_col)
                        
                        # 사용자 정의 지표 계산
                        if custom_indicators:
                            df_filtered = calculate_custom_indicators(df_filtered, custom_indicators)
                        
                        # 해당 날짜의 데이터 추출
                        df_today = df_filtered[df_filtered[date_col] == pd.to_datetime(trading_date)]
                        
                        print(f"해당 날짜 데이터 추출 - {code}: {trading_date}, {len(df_today)}행")
                        
                        # 점수 계산 부분 수정 (전체 종목 데이터를 한 번에 처리)
                        if len(df_today) > 0:
                            row = df_today.iloc[0]
                            
                            # 점수 계산 (기본적으로 활성화)
                            score = None
                            try:
                                # 현재 행을 DataFrame으로 변환
                                df_score = pd.DataFrame([row])
                                
                                # 디버깅: 필요한 컬럼들이 있는지 확인
                                required_features = list(scoring_config.get("numeric", {}).keys())
                                available_features = [feat for feat in required_features if feat in df_score.columns]
                                missing_features = [feat for feat in required_features if feat not in df_score.columns]
                                
                                # 각 피처의 값 확인
                                print(f"\n=== Debug for {code} ===")
                                print(f"Available features: {available_features}")
                                print(f"Missing features: {missing_features}")
                                
                                for feat in required_features:
                                    if feat in df_score.columns:
                                        value = df_score[feat].iloc[0]
                                        print(f"{feat}: {value} (type: {type(value)})")
                                    else:
                                        print(f"{feat}: NOT FOUND")
                                
                                # 사용 가능한 피처만으로 설정 수정
                                if available_features:
                                    # 동적으로 설정 생성
                                    dynamic_config = {
                                        "numeric": {feat: scoring_config.get("numeric", {}).get(feat, {"weight": 0.1, "norm": "rank", "higher_is_better": True}) for feat in available_features},
                                        "rules": scoring_config.get("rules", []),
                                        "filters": scoring_config.get("filters", {})
                                    }
                                    
                                    # 점수 계산 (전체 종목 데이터를 한 번에 처리하도록 수정)
                                    score, contrib = score_frame_single_row(df_score, dynamic_config, all_results)
                                    
                                    if len(score) > 0:
                                        score = score.iloc[0]
                                        print(f"Score calculated for {code}: {score:.2f} (using features: {available_features})")
                                    else:
                                        print(f"No score calculated for {code} - empty result")
                                        score = None
                                else:
                                    print(f"No available features for {code}")
                                    score = None
                                    
                            except Exception as e:
                                print(f"Score calculation error for {code}: {e}")
                                import traceback
                                traceback.print_exc()
                                score = None
                            
                            result = {
                                'date': trading_date,
                                'code': code,
                                'name': CODE_TO_NAME.get(code, code),
                                'close': row[close_col],
                                'return_2d': row.get('return_2d', None),
                                'return_20d': row.get('return_20d', None),
                                'return_60d': row.get('return_60d', None),
                                'sma5': row.get('sma5', None),
                                'sma10': row.get('sma10', None),
                                'sma20': row.get('sma20', None),
                                'sma60': row.get('sma60', None),
                                'sma120': row.get('sma120', None),
                                'ema12': row.get('ema12', None),
                                'ema26': row.get('ema26', None),
                                'rsi': row.get('rsi', None),
                                'macd': row.get('macd', None),
                                'macd_signal': row.get('macd_signal', None),
                                'macd_histogram': row.get('macd_histogram', None),
                                'bb_upper': row.get('bb_upper', None),
                                'bb_middle': row.get('bb_middle', None),
                                'bb_lower': row.get('bb_lower', None),
                                'bb_width': row.get('bb_width', None),
                                'bb_position': row.get('bb_position', None),
                                'volume_sma5': row.get('volume_sma5', None),
                                'volume_sma20': row.get('volume_sma20', None),
                                'volume_ratio': row.get('volume_ratio', None),
                                'atr': row.get('atr', None),
                                'volatility_20': row.get('volatility_20', None),
                                'volatility_60': row.get('volatility_60', None),
                                'volatility_annualized_20': row.get('volatility_annualized_20', None),
                                'volatility_annualized_60': row.get('volatility_annualized_60', None),
                                'parkinson_volatility': row.get('parkinson_volatility', None),
                                'garman_klass_volatility': row.get('garman_klass_volatility', None),
                                'true_range_volatility': row.get('true_range_volatility', None),
                                'rel_mom_20': row.get('rel_mom_20', None),
                                'rel_mom_60': row.get('rel_mom_60', None),
                                'rel_mom_120': row.get('rel_mom_120', None),
                                'high_52w_ratio': row.get('high_52w_ratio', None),
                                'low_52w_ratio': row.get('low_52w_ratio', None),
                                'sharpe_ratio_20': row.get('sharpe_ratio_20', None),
                                'sharpe_ratio_60': row.get('sharpe_ratio_60', None),
                                'sharpe_ratio_120': row.get('sharpe_ratio_120', None),
                                'sortino_ratio_20': row.get('sortino_ratio_20', None),
                                'sortino_ratio_60': row.get('sortino_ratio_60', None),
                                'sortino_ratio_120': row.get('sortino_ratio_120', None),
                                'calmar_ratio_20': row.get('calmar_ratio_20', None),
                                'calmar_ratio_60': row.get('calmar_ratio_60', None),
                                'calmar_ratio_120': row.get('calmar_ratio_120', None),
                                'score': score
                            }
                            
                            daily_results.append(result)
                            print(f"결과 추가 완료 - {code}: {trading_date}")
                            
                except Exception as e:
                    print(f"종목 처리 중 오류 발생 - {code}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            all_results.extend(daily_results)
            print(f"일별 결과 추가 완료 - {trading_date}: {len(daily_results)}개 종목")

            # 진행 상황 완료
            progress_bar.empty()
            status_text.empty()
            
            # 결과를 session_state에 저장
            st.session_state.analysis_results = all_results
            st.session_state.analysis_completed = True
        
        # 결과를 DataFrame으로 변환
        if all_results:
            df_results = pd.DataFrame(all_results)
            st.session_state.analysis_results = df_results.to_dict('records')
            
            # 디버깅 정보 추가
            st.write(f"**분석 완료**: 총 {len(all_results)}개의 결과 생성")
            st.write(f"**데이터프레임 크기**: {df_results.shape}")
            st.write(f"**컬럼 목록**: {list(df_results.columns)}")
            
            # 날짜별로 그룹화하여 표시
            st.subheader("분석 결과")
            
            # 날짜 선택
            unique_dates = sorted(df_results['date'].unique())
            selected_date = st.selectbox(
                "날짜 선택",
                options=unique_dates,
                format_func=lambda x: x.strftime('%Y-%m-%d'),
                index=len(unique_dates)-1  # 기본값으로 마지막 날짜 선택
            )
            
            # 선택된 날짜의 데이터만 필터링
            df_selected = df_results[df_results['date'] == selected_date].copy()
            
            if len(df_selected) > 0:
                # 결과 표시
                st.write(f"**{selected_date.strftime('%Y년 %m월 %d일')} 분석 결과**")
                
                # 컬럼 순서 조정 (점수 추가)
                display_columns = [
                    'name', 'close', 'score',
                    'return_2d', 'return_20d', 'return_60d',
                    'sma5', 'sma10', 'sma20', 'sma60', 'sma120',
                    'ema12', 'ema26', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                    'volume_sma5', 'volume_sma20', 'volume_ratio', 'atr',
                    'volatility_20', 'volatility_60', 'volatility_annualized_20', 'volatility_annualized_60',
                    'parkinson_volatility', 'garman_klass_volatility', 'true_range_volatility',
                    'rel_mom_20', 'rel_mom_60', 'rel_mom_120',
                    'high_52w_ratio', 'low_52w_ratio',
                    'sharpe_ratio_20', 'sharpe_ratio_60', 'sharpe_ratio_120',
                    'sortino_ratio_20', 'sortino_ratio_60', 'sortino_ratio_120',
                    'calmar_ratio_20', 'calmar_ratio_60', 'calmar_ratio_120'
                ]
                
                # 컬럼명 한글로 변경 (점수 추가)
                column_mapping = {
                    'name': '종목명',
                    'close': '종가',
                    'score': '종합점수',
                    'return_2d': '2일 상승률(%)',
                    'return_20d': '20일 상승률(%)',
                    'return_60d': '60일 상승률(%)',
                    'sma5': 'SMA5',
                    'sma10': 'SMA10',
                    'sma20': 'SMA20',
                    'sma60': 'SMA60',
                    'sma120': 'SMA120',
                    'ema12': 'EMA12',
                    'ema26': 'EMA26',
                    'rsi': 'RSI',
                    'macd': 'MACD',
                    'macd_signal': 'MACD Signal',
                    'macd_histogram': 'MACD Histogram',
                    'bb_upper': 'BB Upper',
                    'bb_middle': 'BB Middle',
                    'bb_lower': 'BB Lower',
                    'bb_width': 'BB Width(%)',
                    'bb_position': 'BB Position(%)',
                    'volume_sma5': 'Volume SMA5',
                    'volume_sma20': 'Volume SMA20',
                    'volume_ratio': 'Volume Ratio',
                    'atr': 'ATR',
                    'volatility_20': '변동성 20일(%)',
                    'volatility_60': '변동성 60일(%)',
                    'volatility_annualized_20': '연율변동성 20일(%)',
                    'volatility_annualized_60': '연율변동성 60일(%)',
                    'parkinson_volatility': 'Parkinson 변동성(%)',
                    'garman_klass_volatility': 'Garman-Klass 변동성(%)',
                    'true_range_volatility': 'True Range 변동성(%)',
                    'rel_mom_20': '20일 상대모멘텀(%)',
                    'rel_mom_60': '60일 상대모멘텀(%)',
                    'rel_mom_120': '120일 상대모멘텀(%)',
                    'high_52w_ratio': '52주고점비율(%)',
                    'low_52w_ratio': '52주저점비율(%)',
                    'sharpe_ratio_20': 'Sharpe 지수 20일',
                    'sharpe_ratio_60': 'Sharpe 지수 60일',
                    'sharpe_ratio_120': 'Sharpe 지수 120일',
                    'sortino_ratio_20': 'Sortino 지수 20일',
                    'sortino_ratio_60': 'Sortino 지수 60일',
                    'sortino_ratio_120': 'Sortino 지수 120일',
                    'calmar_ratio_20': 'Calmar 지수 20일',
                    'calmar_ratio_60': 'Calmar 지수 60일',
                    'calmar_ratio_120': 'Calmar 지수 120일'
                }
                
                df_display = df_selected[display_columns].copy()
                df_display.columns = [column_mapping[col] for col in display_columns]
                
                # 소수점 자리수 조정
                numeric_columns = [col for col in df_display.columns if col != '종목명']
                df_display[numeric_columns] = df_display[numeric_columns].round(2)
                
                # 종가를 정수로 표시
                df_display['종가'] = df_display['종가'].astype(int)
                
                st.dataframe(df_display, use_container_width=True)
                
                # 전체 기간 데이터 다운로드 버튼
                st.subheader("전체 데이터 다운로드")
                csv_data = df_results.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="CSV 파일 다운로드",
                    data=csv_data,
                    file_name=f"stock_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("선택된 날짜에 데이터가 없습니다.")
        else:
            st.warning("분석 결과가 없습니다.")
else:
    st.info("분석할 종목을 선택하고 '분석 시작' 버튼을 클릭하세요.")



# ==============================
# 기본 성능 검증: 점수 구간별 다음날 수익률
# ==============================
def _render_score_bin_eval():
    st.subheader("점수 구간별 수익률 비교 분석")

    # Try to find results
    df_results = None
    if 'df_results' in globals():
        try:
            if isinstance(globals()['df_results'], (pd.DataFrame, pd.core.frame.DataFrame)):
                df_results = globals()['df_results']
        except Exception:
            pass
    if df_results is None:
        # Try from session_state
        try:
            if st.session_state.get('analysis_results') is not None:
                df_results = pd.DataFrame(st.session_state.analysis_results)
        except Exception:
            df_results = None

    if df_results is None or df_results.empty or 'score' not in df_results.columns or 'close' not in df_results.columns:
        st.info("검증을 위한 결과 데이터가 없습니다. (필수 컬럼: score, close, date, code)")
        return

    df_eval = df_results.copy()
    # Ensure required columns exist
    required_cols = {'date','code','close','score'}
    missing = required_cols - set(map(str.lower, df_eval.columns))
    # Try to normalize column names to lower-case if needed
    df_eval.columns = [c.lower() for c in df_eval.columns]
    required_cols = {'date','code','close','score'}
    if not required_cols.issubset(df_eval.columns):
        st.warning(f"검증 불가 - 필요한 컬럼 누락: {required_cols - set(df_eval.columns)}")
        return

    # Prepare
    df_eval['date'] = pd.to_datetime(df_eval['date'])
    df_eval = df_eval.sort_values(['code','date'])
    
    # 여러 기간의 수익률 계산
    df_eval['close_next_1d'] = df_eval.groupby('code')['close'].shift(-1)
    df_eval['close_next_3d'] = df_eval.groupby('code')['close'].shift(-3)
    df_eval['close_next_5d'] = df_eval.groupby('code')['close'].shift(-5)
    df_eval['close_next_20d'] = df_eval.groupby('code')['close'].shift(-20)
    
    df_eval['ret_next_1d'] = (df_eval['close_next_1d'] / df_eval['close'] - 1.0) * 100.0
    df_eval['ret_next_3d'] = (df_eval['close_next_3d'] / df_eval['close'] - 1.0) * 100.0
    df_eval['ret_next_5d'] = (df_eval['close_next_5d'] / df_eval['close'] - 1.0) * 100.0
    df_eval['ret_next_20d'] = (df_eval['close_next_20d'] / df_eval['close'] - 1.0) * 100.0

    # 결측치 처리
    df_eval = df_eval.replace([np.inf, -np.inf], np.nan).dropna(subset=['score','ret_next_1d'])

    q = st.slider("분위수 구간 수(q)", min_value=3, max_value=10, value=5, step=1, key="q_bins_score_eval")
    try:
        df_eval['score_bin'] = pd.qcut(df_eval['score'], q=q, labels=[f"Q{i}" for i in range(1, q+1)])
    except Exception:
        st.warning("스코어 분포가 치우쳐 분위수 구간을 만들기 어렵습니다. q 값을 줄여보세요.")
        df_eval['score_bin'] = pd.qcut(df_eval['score'], q=3, labels=["Q1","Q2","Q3"])
        q = 3

    # 각 기간별 성과 분석
    periods = [
        ('ret_next_1d', '다음날'),
        ('ret_next_3d', '3일 뒤'),
        ('ret_next_5d', '5거래일 뒤'),
        ('ret_next_20d', '20거래일 뒤')
    ]
    
    for ret_col, period_name in periods:
        st.write(f"**{period_name} 수익률 분석**")
        
        bin_perf = (df_eval.groupby('score_bin')
                    .agg(mean_ret=(ret_col,'mean'),
                         median_ret=(ret_col,'median'),
                         win_rate=(ret_col, lambda x: np.mean(x > 0) * 100.0),
                         count=(ret_col,'size'),
                         mean_score=('score','mean'))
                    .reset_index()
                    .sort_values('score_bin'))

        try:
            st.dataframe(bin_perf.style.format({
                'mean_ret': '{:.3f}%',
                'median_ret': '{:.3f}%',
                'win_rate': '{:.1f}%',
                'mean_score': '{:.2f}'
            }), use_container_width=True)
        except Exception:
            st.dataframe(bin_perf, use_container_width=True)

        # Top - Bottom spread
        top_label, bottom_label = f"Q{q}", "Q1"
        try:
            top_val = bin_perf.loc[bin_perf['score_bin'].astype(str)==top_label, 'mean_ret'].values[0]
            bot_val = bin_perf.loc[bin_perf['score_bin'].astype(str)==bottom_label, 'mean_ret'].values[0]
            spread = top_val - bot_val
            st.metric(f"{top_label} - {bottom_label} 평균 {period_name} 수익률 차이", f"{spread:.3f}%")
        except Exception:
            pass
        
        st.write("---")

    # Daily IC (Spearman) - scipy 없이 계산
    st.write("**IC (Information Coefficient) 분석**")
    try:
        # 더 안전한 IC 계산 방법 (scipy 없이)
        daily_ic_list = []
        total_dates = 0
        
        for date in df_eval['date'].unique():
            total_dates += 1
            date_data = df_eval[df_eval['date'] == date]
            
            if len(date_data) > 1:  # 최소 2개 이상의 데이터가 있어야 상관관계 계산 가능
                try:
                    # 안전한 Spearman 상관관계 계산
                    ic = safe_spearman_corr(date_data['score'], date_data['ret_next_1d'])
                    if not pd.isna(ic) and abs(ic) <= 1.0:  # 유효한 상관관계 값인지 확인
                        daily_ic_list.append(ic)
                except Exception as e:
                    print(f"IC 계산 오류 (날짜: {date}): {e}")
                    continue
        
        if daily_ic_list:
            daily_ic = pd.Series(daily_ic_list)
            st.metric("일별 평균 IC (Spearman)", f"{daily_ic.mean():.3f}")
            st.write(f"IC 표준편차: {daily_ic.std():.3f}, 관측치 수: {len(daily_ic)}")
            st.write(f"전체 날짜 수: {total_dates}, IC 계산 성공: {len(daily_ic_list)}")
        else:
            st.warning("IC 계산을 위한 충분한 데이터가 없습니다.")
            st.write(f"전체 날짜 수: {total_dates}, IC 계산 성공: 0")
    except Exception as e:
        st.warning(f"IC 계산 중 오류: {e}")
        st.write(f"오류 상세: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    # 분석 결과 설명
    st.write("---")
    st.subheader("📊 분석 결과 해석 가이드")
    
    st.write("""
    ### 🔍 **점수 구간별 수익률 비교 분석**
    
    **분위수(Q) 구분:**
    - **Q1**: 가장 낮은 점수 구간 (0-20% 또는 0-25%)
    - **Q2~Q4**: 중간 점수 구간들
    - **Q5**: 가장 높은 점수 구간 (80-100% 또는 75-100%)
    
    **지표 해석:**
    - **평균 수익률**: 해당 구간의 평균 수익률
    - **중간값 수익률**: 중간값 기준 수익률 (극단값 영향 적음)
    - **승률**: 양수 수익률을 기록한 비율
    - **평균 점수**: 해당 구간의 평균 스코어링 점수
    
    **성과 평가:**
    - **Q5 - Q1 차이**: 높은 점수 구간과 낮은 점수 구간의 수익률 차이
    - **양수 차이**: 스코어링 시스템이 유효함을 의미
    - **차이가 클수록**: 더 강력한 예측력을 가짐
    
    ### 📈 **IC (Information Coefficient) 분석**
    
    **IC란?**
    - 점수와 실제 수익률 간의 순위 상관관계
    - -1 ~ +1 범위 (0에 가까울수록 예측력 없음)
    - **양수**: 높은 점수 → 높은 수익률 (정상)
    - **음수**: 높은 점수 → 낮은 수익률 (역방향)
    
    **IC 해석 기준:**
    - **0.05 이상**: 좋은 예측력
    - **0.10 이상**: 매우 좋은 예측력
    - **0.15 이상**: 탁월한 예측력
    
    **IC 표준편차:**
    - 낮을수록 안정적인 예측력
    - 높을수록 변동성이 큰 예측력
    
    ### 🎯 **투자 전략 활용**
    
    **단기 전략 (1-3일):**
    - 높은 IC와 큰 Q5-Q1 차이 확인
    - 단기 모멘텀 활용
    
    **중기 전략 (5-20일):**
    - 지속적인 예측력 확인
    - 포트폴리오 리밸런싱 주기 결정
    
    **리스크 관리:**
    - IC 표준편차가 높으면 보수적 접근
    - 승률과 함께 종합 판단
    """)

# 자동 렌더: 페이지 하단에서 한 번 호출
try:
    _render_score_bin_eval()
except Exception:
    # If Streamlit context not ready, ignore
    pass