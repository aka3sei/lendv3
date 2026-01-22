import streamlit as st
import pickle
import pandas as pd
import re

# --- 関数: 学習地点の抽出 (念のため保持) ---
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
    
    # セレクトボックス用のリスト作成
    all_points = sorted(point_mean_dict.keys())
    # 「東京都千代田区一番町」から「千代田区」を抽出してユニークなリストを作る
    wards = sorted(list(set([re.search(r'東京都(.+?区)', p).group(1) for p in all_points if '区' in p])))
    
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# --- UI部分 ---
st.title("AI賃料査定アプリ (23区版)")

# 入力エリア
st.sidebar.header("物件情報入力")

# 1. 区を選択
selected_ward = st.sidebar.selectbox("区を選択", wards)

# 2. 選択された区に属する「学習地点」を抽出してリスト化
filtered_points = [p for p in all_points if selected_ward in p]
selected_point = st.sidebar.selectbox("所在地 (町丁目)", filtered_points)

# 3. その他の数値入力
area = st.sidebar.number_input("専有面積 (㎡)", min_value=10.0, max_value=200.0, value=25.0, step=0.1)
built_year = st.sidebar.number_input("築年 (西暦)", min_value=1950, max_value=2026, value=2015)
walk_dist = st.sidebar.number_input("駅徒歩 (分)", min_value=0, max_value=30, value=5)

if st.sidebar.button("査定実行"):
    # 特徴量の準備
    unit_price = point_mean_dict.get(selected_point, global_mean)
    age = 2026 - built_year
    
    # 予測実行
    X_input = pd.DataFrame([[unit_price, area, age, walk_dist]], 
                           columns=['地点平均単価', '専有面積(㎡)', '築年数', '駅徒歩(分)'])
    prediction = model.predict(X_input)[0]
    
    # --- メイン画面表示 ---
    st.metric(label="AI予測賃料", value=f"{int(prediction):,} 円")
    
    # サンプル数(SP)の表示
    sp_count = point_count_dict.get(selected_point, 0)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**【査定根拠データ】**")
        st.write(f"選択地点: {selected_point}")
        st.write(f"SP: **{sp_count} **")
    with col2:
        st.write("**【物件スペック】**")
        st.write(f"築年数: {age} 年")
        st.write(f"駅徒歩: {walk_dist} 分")

else:
    st.info("左側のサイドバーで条件を選択し、「査定実行」ボタンを押してください。")
