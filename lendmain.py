import streamlit as st
import pickle
import pandas as pd
import re

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
    
    # 全地点リスト
    all_full_points = sorted(point_mean_dict.keys())
    
    # 「区」のリスト作成
    wards = sorted(list(set([re.search(r'東京都(.+?区)', p).group(1) for p in all_full_points if '区' in p])))
    
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# --- UI部分 ---
st.title("AI賃料査定")

# 1. 区を選択
selected_ward = st.selectbox("区を選択", wards)

# 2. 所在地（町丁目）を選択：表示から「東京都〇〇区」を消す
# 例：「東京都世田谷区三宿1丁目」 -> 「三宿1丁目」
prefix = f"東京都{selected_ward}"
relevant_full_points = [p for p in all_full_points if p.startswith(prefix)]
display_points = [p.replace(prefix, "") for p in relevant_full_points]

selected_display_point = st.selectbox("所在地 (町丁目)", display_points)
# 予測用に元のフル住所を復元
selected_full_point = prefix + selected_display_point

# 3. スペック入力
col1, col2, col3 = st.columns(3)
with col1:
    area = st.number_input("専有面積 (㎡)", min_value=10.0, max_value=200.0, value=25.0, step=0.1)
with col2:
    built_year = st.number_input("築年 (西暦)", min_value=1950, max_value=2026, value=2015)
with col3:
    walk_dist = st.number_input("駅徒歩 (分)", min_value=0, max_value=30, value=5)

if st.button("査定実行", use_container_width=True):
    # 特徴量の準備
    unit_price = point_mean_dict.get(selected_full_point, global_mean)
    age = 2026 - built_year
    
    # 予測実行
    X_input = pd.DataFrame([[unit_price, area, age, walk_dist]], 
                           columns=['地点平均単価', '専有面積(㎡)', '築年数', '駅徒歩(分)'])
    prediction = model.predict(X_input)[0]
    
    # --- 結果表示 ---
    st.divider()
    st.metric(label="AI予測賃料", value=f"{int(prediction):,} 円")
    
    # SP（サンプル数）の表示
    sp_count = point_count_dict.get(selected_full_point, 0)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**地点:** {selected_display_point}")
    with c2:
        st.write(f"**SP:** {sp_count}")
