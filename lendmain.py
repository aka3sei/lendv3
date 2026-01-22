import streamlit as st
import pickle
import pandas as pd
import re

# --- 関数: 学習地点の抽出 (学習時と同じロジック) ---
def extract_learning_point(address):
    if not isinstance(address, str): return "Unknown"
    match = re.search(r'(.+?[区市])(.+?\d+丁目)', address)
    if match: return match.group(1) + match.group(2)
    match_no_chome = re.search(r'(.+?[区市])([^0-9]+)', address)
    if match_no_chome: return match_no_chome.group(1) + match_no_chome.group(2)
    return address[:15]

# --- モデルの読み込み ---
@st.cache_resource
def load_model():
    with open('rent_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    data = load_model()
    model = data['model']
    point_mean_dict = data['point_mean']
    point_count_dict = data['point_count']
    global_mean = data['global_mean_unit_price']
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# --- UI部分 ---
st.title("AI賃料査定アプリ")
st.write("物件情報を入力すると、AIが予測賃料を算出します。")

col1, col2 = st.columns(2)

with col1:
    address = st.text_input("所在地", placeholder="東京都中央区銀座1丁目")
    area = st.number_input("専有面積 (㎡)", min_value=10.0, max_value=200.0, value=25.0, step=0.1)

with col2:
    built_year = st.number_input("築年 (西暦)", min_value=1950, max_value=2026, value=2015)
    walk_dist = st.number_input("駅徒歩 (分)", min_value=0, max_value=30, value=5)

if st.button("査定する"):
    # 1. 入力住所から学習地点を特定
    lp = extract_learning_point(address)
    
    # 2. 特徴量の準備
    # 地点単価を取得（未知の地点の場合は全体平均を使用）
    unit_price = point_mean_dict.get(lp, global_mean)
    age = 2026 - built_year
    
    # 3. 予測実行
    X_input = pd.DataFrame([[unit_price, area, age, walk_dist]], 
                           columns=['地点平均単価', '専有面積(㎡)', '築年数', '駅徒歩(分)'])
    prediction = model.predict(X_input)[0]
    
    # --- 結果表示 ---
    st.divider()
    st.subheader(f"予測賃料: {int(prediction):,} 円")
    
    # --- サンプル数(SP)の表示 ---
    sp_count = point_count_dict.get(lp, 0)
    
    # 下の方に控えめに表示
    st.caption(f"【査定データ詳細】")
    st.write(f"学習地点: {lp}")
    st.write(f"近隣サンプル数 (SP): {sp_count} 件")
    
    if sp_count == 0:
        st.warning("※入力された地点の直接的な学習データがないため、周辺相場から推計しています。")
    elif sp_count < 5:
        st.info("※サンプル数が少ないため、参考値としてご確認ください。")