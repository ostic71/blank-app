import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ─── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Linear vs Huber Regression", layout="wide")

# Ẩn menu & footer
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Phần 1: 3D Visualization (GaltonFamilies Dataset with Outliers) ───────────

# Tải dataset GaltonFamilies
galton = pd.read_csv("family.csv")

# Trích xuất features và target
X = galton[['father', 'mother']].values
y = galton['childHeight'].values

# Thêm outliers nhân tạo để làm nổi bật sự khác biệt
np.random.seed(42)
outlier_indices = np.random.choice(len(y), 0, replace=False)  # Không thêm outliers
y_with_outliers = y.copy()
# y_with_outliers[outlier_indices] += np.random.uniform(-200, -180, 0)  # Không thêm outliers

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Huấn luyện Linear Regression
lr = LinearRegression()
lr.fit(X_scaled, y_with_outliers)

# Huấn luyện Huber Regression
huber = HuberRegressor()
huber.fit(X_scaled, y_with_outliers)

# Đánh giá mô hình với outliers
y_lr_pred_with_outliers = lr.predict(X_scaled)
y_huber_pred_with_outliers = huber.predict(X_scaled)
mse_lr_with_outliers = mean_squared_error(y_with_outliers, y_lr_pred_with_outliers)
r2_lr_with_outliers = r2_score(y_with_outliers, y_lr_pred_with_outliers)
mse_huber_with_outliers = mean_squared_error(y_with_outliers, y_huber_pred_with_outliers)
r2_huber_with_outliers = r2_score(y_with_outliers, y_huber_pred_with_outliers)

# Tạo lưới cho mặt phẳng hồi quy
x1_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 20)
x2_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Dự đoán cho Linear Regression
y_lr_pred = lr.predict(X_grid)
y_lr_grid = y_lr_pred.reshape(x1_grid.shape)

# Dự đoán cho Huber Regression
y_huber_pred = huber.predict(X_grid)
y_huber_grid = y_huber_pred.reshape(x1_grid.shape)

# Tạo biểu đồ 3D tương tác với Plotly
fig_3d_with_outliers = go.Figure()

# Vẽ điểm dữ liệu
fig_3d_with_outliers.add_trace(go.Scatter3d(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    z=y_with_outliers,
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.5),
    name='Dữ liệu (không có outliers)'
))

# Vẽ mặt phẳng Linear Regression
fig_3d_with_outliers.add_trace(go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=y_lr_grid,
    colorscale='Reds',
    opacity=0.4,
    name='Linear Regression',
    showscale=False
))

# Vẽ mặt phẳng Huber Regression
fig_3d_with_outliers.add_trace(go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=y_huber_grid,
    colorscale='Greens',
    opacity=0.4,
    name='Huber Regression',
    showscale=False
))

# Thiết lập layout
fig_3d_with_outliers.update_layout(
    title='Linear vs Huber Regression trong Không Gian 3D (Không Có Outliers)',
    scene=dict(
        xaxis_title='Father Height (chuẩn hóa)',
        yaxis_title='Mother Height (chuẩn hóa)',
        zaxis_title='Child Height',
    ),
    legend=dict(x=0.1, y=0.9),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1000,
    height=600
)

# ─── Phần 2: 3D Visualization (GaltonFamilies Dataset with Outliers) ────────

# Thêm outliers
np.random.seed(42)
outlier_indices = np.random.choice(len(y), 100, replace=False)  # 100 outliers
y_with_outliers_100 = y.copy()
y_with_outliers_100[outlier_indices] += np.random.uniform(-200, -180, 100)  # Outliers từ -200 đến -180 inches

# Huấn luyện Linear Regression
lr_no_outliers = LinearRegression()
lr_no_outliers.fit(X_scaled, y_with_outliers_100)

# Huấn luyện Huber Regression
huber_no_outliers = HuberRegressor()
huber_no_outliers.fit(X_scaled, y_with_outliers_100)

# Đánh giá mô hình không có outliers
y_lr_pred_no_outliers = lr_no_outliers.predict(X_scaled)
y_huber_pred_no_outliers = huber_no_outliers.predict(X_scaled)
mse_lr_no_outliers = mean_squared_error(y_with_outliers_100, y_lr_pred_no_outliers)
r2_lr_no_outliers = r2_score(y_with_outliers_100, y_lr_pred_no_outliers)
mse_huber_no_outliers = mean_squared_error(y_with_outliers_100, y_huber_pred_no_outliers)
r2_huber_no_outliers = r2_score(y_with_outliers_100, y_huber_pred_no_outliers)

# Dự đoán cho Linear Regression
y_lr_pred_no_outliers = lr_no_outliers.predict(X_grid)
y_lr_grid_no_outliers = y_lr_pred_no_outliers.reshape(x1_grid.shape)

# Dự đoán cho Huber Regression
y_huber_pred_no_outliers = huber_no_outliers.predict(X_grid)
y_huber_grid_no_outliers = y_huber_pred_no_outliers.reshape(x1_grid.shape)

# Tạo biểu đồ 3D tương tác với Plotly
fig_3d_no_outliers = go.Figure()

# Vẽ điểm dữ liệu
fig_3d_no_outliers.add_trace(go.Scatter3d(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    z=y_with_outliers_100,
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.5),
    name='Dữ liệu (có outliers)'
))

# Vẽ mặt phẳng Linear Regression
fig_3d_no_outliers.add_trace(go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=y_lr_grid_no_outliers,
    colorscale='Reds',
    opacity=0.4,
    name='Linear Regression',
    showscale=False
))

# Vẽ mặt phẳng Huber Regression
fig_3d_no_outliers.add_trace(go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=y_huber_grid_no_outliers,
    colorscale='Greens',
    opacity=0.4,
    name='Huber Regression',
    showscale=False
))

# Thiết lập layout
fig_3d_no_outliers.update_layout(
    title='Linear vs Huber Regression trong Không Gian 3D (Có Outliers)',
    scene=dict(
        xaxis_title='Father Height (chuẩn hóa)',
        yaxis_title='Mother Height (chuẩn hóa)',
        zaxis_title='Child Height',
    ),
    legend=dict(x=0.1, y=0.9),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1000,
    height=600
)

# ─── Phần 3: 2D Interactive Plot ──────────────────────────────────────────────

# Khởi tạo dữ liệu cho biểu đồ 2D
np.random.seed(42)
x_init = np.linspace(0, 9, 10)
y_init = 2 * x_init + 1 + np.random.uniform(-0.5, 0.5, 10)

if "data" not in st.session_state:
    st.session_state.data = np.column_stack((x_init, y_init))

# Hàm huấn luyện & dự đoán
def fit_models(data):
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    lin = LinearRegression().fit(X, y)
    huber = HuberRegressor().fit(X, y)
    return lin, huber

# Hàm vẽ biểu đồ 2D
def make_plot(data, lin, huber):
    fig = go.Figure()

    # Scatter dữ liệu
    fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1],
                             mode="markers", name="Dữ liệu"))

    # Đường hồi quy
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

# Hiển thị biểu đồ 3D không có outliers
st.subheader("Biểu đồ 3D: Dự đoán chiều cao con dựa trên chiều cao bố mẹ (Không Có Outliers)")
st.write("Biểu đồ này sử dụng dataset GaltonFamilies không có outliers để so sánh Linear và Huber Regression.")
st.plotly_chart(fig_3d_with_outliers, use_container_width=True)

# Hiển thị hệ số và đánh giá mô hình không có outliers
st.write(f"**Linear Regression Evaluation (Không Có Outliers):** MSE={mse_lr_with_outliers:.2f}, R²={r2_lr_with_outliers:.2f}")
st.write(f"**Huber Regression Evaluation (Không Có Outliers):** MSE={mse_huber_with_outliers:.2f}, R²={r2_huber_with_outliers:.2f}")

# Hiển thị biểu đồ 3D có outliers
st.subheader("Biểu đồ 3D: Dự đoán chiều cao con dựa trên chiều cao bố mẹ (Có Outliers)")
st.write("Biểu đồ này sử dụng dataset GaltonFamilies với 100 outliers nhân tạo để so sánh Linear và Huber Regression.")
st.plotly_chart(fig_3d_no_outliers, use_container_width=True)

# Hiển thị hệ số và đánh giá mô hình có outliers
st.write(f"**Linear Regression Evaluation (Có Outliers):** MSE={mse_lr_no_outliers:.2f}, R²={r2_lr_no_outliers:.2f}")
st.write(f"**Huber Regression Evaluation (Có Outliers):** MSE={mse_huber_no_outliers:.2f}, R²={r2_huber_no_outliers:.2f}")

# Hiển thị biểu đồ 2D
st.subheader("Biểu đồ 2D: So sánh Linear và Huber Regression với dữ liệu tùy chỉnh")
st.write(
    "Linear Regression nhạy cảm với điểm ngoại lệ, còn Huber Regression thì "
    "mạnh mẽ hơn. Thêm điểm mới để xem sự khác biệt!"
)

lin_model, huber_model = fit_models(st.session_state.data)
st.plotly_chart(
    make_plot(st.session_state.data, lin_model, huber_model),
    use_container_width=True
)

# Thêm điểm ngoại lệ
st.subheader("Thêm điểm ngoại lệ")
col1, col2 = st.columns(2)
with col1:
    x_new = st.number_input("Tọa độ X", 0.0, 10.0, 5.0, 0.1)
with col2:
    y_new = st.number_input("Tọa độ Y", 0.0, 20.0, 10.0, 0.1)

if st.button("Thêm điểm"):
    st.session_state.data = np.vstack((st.session_state.data, [x_new, y_new]))
    st.success(f"Đã thêm điểm ({x_new}, {y_new})")
    st.rerun()  # Tự refresh giao diện

# Hiển thị bảng dữ liệu
st.subheader("Danh sách điểm")
st.dataframe(st.session_state.data, use_container_width=True, hide_index=True)