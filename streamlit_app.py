import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, HuberRegressor

# ─── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Linear vs Huber Regression", layout="wide")

# Ẩn menu & footer (bỏ nếu không cần)
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Khởi tạo dữ liệu ──────────────────────────────────────────────────────────
np.random.seed(42)
x_init = np.linspace(0, 9, 10)
y_init = 2 * x_init + 1 + np.random.uniform(-0.5, 0.5, 10)

if "data" not in st.session_state:
    st.session_state.data = np.column_stack((x_init, y_init))

# ─── Hàm huấn luyện & dự đoán ─────────────────────────────────────────────────
def fit_models(data):
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    lin   = LinearRegression().fit(X, y)
    huber = HuberRegressor().fit(X, y)
    return lin, huber

# ─── Hàm vẽ biểu đồ ────────────────────────────────────────────────────────────
def make_plot(data, lin, huber):
    fig = go.Figure()

    # scatter dữ liệu
    fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1],
                             mode="markers", name="Dữ liệu"))

    # đường hồi quy
    x_range = np.linspace(0, 10, 100).reshape(-1, 1)
    fig.add_trace(go.Scatter(x=x_range.flatten(), y=lin.predict(x_range),
                             mode="lines", name="Linear Regression"))
    fig.add_trace(go.Scatter(x=x_range.flatten(), y=huber.predict(x_range),
                             mode="lines", name="Huber Regression"))

    fig.update_layout(title="So sánh Linear Regression và Huber Regression",
                      xaxis_title="X", yaxis_title="Y", legend_title_text="")
    return fig

# ─── Giao diện chính ───────────────────────────────────────────────────────────
st.title("So sánh Linear Regression và Huber Regression")
st.write(
    "Linear Regression nhạy cảm với điểm ngoại lệ, còn Huber Regression thì "
    "mạnh mẽ hơn. Thêm điểm mới để xem sự khác biệt!"
)

lin_model, huber_model = fit_models(st.session_state.data)
st.plotly_chart(
    make_plot(st.session_state.data, lin_model, huber_model),
    use_container_width=True
)

# ─── Thêm điểm ngoại lệ ────────────────────────────────────────────────────────
st.subheader("Thêm điểm ngoại lệ")
col1, col2 = st.columns(2)
with col1:
    x_new = st.number_input("Tọa độ X", 0.0, 10.0, 5.0, 0.1)
with col2:
    y_new = st.number_input("Tọa độ Y", 0.0, 20.0, 10.0, 0.1)

if st.button("Thêm điểm"):
    st.session_state.data = np.vstack((st.session_state.data, [x_new, y_new]))
    st.success(f"Đã thêm điểm ({x_new}, {y_new})")
    st.rerun()   # tự refresh giao diện

# ─── Hiển thị bảng dữ liệu ─────────────────────────────────────────────────────
st.subheader("Danh sách điểm")
st.dataframe(st.session_state.data, use_container_width=True, hide_index=True)
