import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

# ─── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Linear vs Huber Regression", layout="wide")

# Ẩn menu & footer và thêm CSS để tăng kích thước container biểu đồ
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer    {visibility: hidden;}
        .plotly-chart-container {
            width: 100% !important;
            max-width: 1200px !important;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Phần 1: Chuẩn bị dữ liệu GaltonFamilies ───────────────────────────────────

# Tải dataset GaltonFamilies
galton = pd.read_csv("family.csv")

# Mã hóa gender (male=1, female=0)
galton['gender'] = galton['gender'].map({'male': 1, 'female': 0})

# Chọn các thuộc tính để huấn luyện (loại bỏ 'family' và 'midparentHeight')
features = ['father', 'mother', 'children', 'childNum', 'gender']
X = galton[features].values
y = galton['childHeight'].values

# Tách tập train và test (80% train, 20% test)
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu với RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Giảm chiều dữ liệu bằng PCA cho biểu đồ 3D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ─── Phần 2: 3D Visualization (GaltonFamilies Dataset without Outliers) ────────

# Không thêm outliers vào tập train
y_train_no_outliers = y_train.copy()

# Huấn luyện Linear Regression
lr_no_outliers = LinearRegression()
lr_no_outliers.fit(X_train_scaled, y_train_no_outliers)
lr_no_outliers_pred_train = lr_no_outliers.predict(X_train_scaled)
lr_no_outliers_pred_test = lr_no_outliers.predict(X_test_scaled)
lr_no_outliers_mse_train = mean_squared_error(y_train_no_outliers, lr_no_outliers_pred_train)
lr_no_outliers_mse_test = mean_squared_error(y_test, lr_no_outliers_pred_test)

# Huấn luyện Huber Regression với epsilon nhỏ hơn
huber_no_outliers = HuberRegressor(epsilon=1.1)
huber_no_outliers.fit(X_train_scaled, y_train_no_outliers)
huber_no_outliers_pred_train = huber_no_outliers.predict(X_train_scaled)
huber_no_outliers_pred_test = huber_no_outliers.predict(X_test_scaled)
huber_no_outliers_mse_train = mean_squared_error(y_train_no_outliers, huber_no_outliers_pred_train)
huber_no_outliers_mse_test = mean_squared_error(y_test, huber_no_outliers_pred_test)

# Tạo lưới cho mặt phẳng hồi quy (dựa trên PCA components)
x1_range = np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 20)
x2_range = np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid_pca = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Chuyển ngược PCA grid về không gian gốc để dự đoán
X_grid_scaled = pca.inverse_transform(X_grid_pca)
y_lr_pred_no_outliers = lr_no_outliers.predict(X_grid_scaled)
y_lr_grid_no_outliers = y_lr_pred_no_outliers.reshape(x1_grid.shape)
y_huber_pred_no_outliers = huber_no_outliers.predict(X_grid_scaled)
y_huber_grid_no_outliers = y_huber_pred_no_outliers.reshape(x1_grid.shape)

# Tạo biểu đồ 3D không có outliers
fig_3d_no_outliers = go.Figure()

# Vẽ điểm dữ liệu (train và test)
fig_3d_no_outliers.add_trace(go.Scatter3d(
    x=X_train_pca[:, 0],
    y=X_train_pca[:, 1],
    z=y_train_no_outliers,
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.5),
    name='Dữ liệu Train (không có outliers)'
))
fig_3d_no_outliers.add_trace(go.Scatter3d(
    x=X_test_pca[:, 0],
    y=X_test_pca[:, 1],
    z=y_test,
    mode='markers',
    marker=dict(size=5, color='orange', opacity=0.5),
    name='Dữ liệu Test'
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
    title='Linear vs Huber Regression trong Không Gian 3D (Không Có Outliers, PCA)',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='Child Height',
    ),
    legend=dict(x=0.1, y=0.9),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1000,
    height=600
)

# ─── Phần 3: 3D Visualization (GaltonFamilies Dataset with Outliers) ───────────

# Thêm outliers nhân tạo vào tập train
np.random.seed(42)
outlier_indices = np.random.choice(len(y_train), int(0.2 * len(y_train)), replace=False)  # 20% train là outliers
y_train_with_outliers = y_train.copy()
y_train_with_outliers[outlier_indices] += np.random.uniform(20, 40, len(outlier_indices))  # Outliers từ 20 đến 40 inches

# Huấn luyện Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_with_outliers)
lr_pred_train = lr.predict(X_train_scaled)
lr_pred_test = lr.predict(X_test_scaled)
lr_mse_train = mean_squared_error(y_train_with_outliers, lr_pred_train)
lr_mse_test = mean_squared_error(y_test, lr_pred_test)

# Huấn luyện Huber Regression với epsilon nhỏ hơn
huber = HuberRegressor(epsilon=1.1)
huber.fit(X_train_scaled, y_train_with_outliers)
huber_pred_train = huber.predict(X_train_scaled)
huber_pred_test = huber.predict(X_test_scaled)
huber_mse_train = mean_squared_error(y_train_with_outliers, huber_pred_train)
huber_mse_test = mean_squared_error(y_test, huber_pred_test)

# Dự đoán cho Linear Regression
y_lr_pred = lr.predict(X_grid_scaled)
y_lr_grid = y_lr_pred.reshape(x1_grid.shape)

# Dự đoán cho Huber Regression
y_huber_pred = huber.predict(X_grid_scaled)
y_huber_grid = y_huber_pred.reshape(x1_grid.shape)

# Tạo biểu đồ 3D với outliers
fig_3d_with_outliers = go.Figure()

# Vẽ điểm dữ liệu (train và test)
fig_3d_with_outliers.add_trace(go.Scatter3d(
    x=X_train_pca[:, 0],
    y=X_train_pca[:, 1],
    z=y_train_with_outliers,
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.5),
    name='Dữ liệu Train (có outliers)'
))
fig_3d_with_outliers.add_trace(go.Scatter3d(
    x=X_test_pca[:, 0],
    y=X_test_pca[:, 1],
    z=y_test,
    mode='markers',
    marker=dict(size=5, color='orange', opacity=0.5),
    name='Dữ liệu Test'
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
    title='Linear vs Huber Regression trong Không Gian 3D (Có Outliers, PCA)',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='Child Height',
    ),
    legend=dict(x=0.1, y=0.9),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1000,
    height=600
)

# ─── Phần 4: 2D Interactive Plot ──────────────────────────────────────────────

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
    huber = HuberRegressor(epsilon=1.1).fit(X, y)
    lin_pred = lin.predict(X)
    huber_pred = huber.predict(X)
    lin_mse = mean_squared_error(y, lin_pred)
    huber_mse = mean_squared_error(y, huber_pred)
    return lin, huber, lin_mse, huber_mse

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
st.subheader("Biểu đồ 3D: Dự đoán chiều cao con (Không Có Outliers)")
st.write(f"Sử dụng các thuộc tính {', '.join(features)} từ dataset GaltonFamilies, giảm chiều bằng PCA. "
         f"Không có outliers trong tập train. Tập test không có outliers.")
st.plotly_chart(fig_3d_no_outliers, use_container_width=True)

# Hiển thị metrics mô hình 3D không có outliers
st.write("**Linear Regression (Không Có Outliers):**")
# st.write(f"Intercept: {lr_no_outliers.intercept_:.2f}")
st.write(f"Train MSE: {lr_no_outliers_mse_train:.2f}")
st.write(f"Test MSE: {lr_no_outliers_mse_test:.2f}")
st.write("**Huber Regression (Không Có Outliers):**")
# st.write(f"Intercept: {huber_no_outliers.intercept_:.2f}")
st.write(f"Train MSE: {huber_no_outliers_mse_train:.2f}")
st.write(f"Test MSE: {huber_no_outliers_mse_test:.2f}")

# Hiển thị biểu đồ 3D với outliers
st.subheader("Biểu đồ 3D: Dự đoán chiều cao con (Có Outliers)")
st.write(f"Sử dụng các thuộc tính {', '.join(features)} từ dataset GaltonFamilies, giảm chiều bằng PCA. "
         f"Tập train có {len(outlier_indices)} outliers (20-40 inches). Tập test không có outliers.")
st.plotly_chart(fig_3d_with_outliers, use_container_width=True)

# Hiển thị metrics mô hình 3D với outliers
st.write("**Linear Regression (Có Outliers):**")
# st.write(f"Intercept: {lr.intercept_:.2f}")
st.write(f"Train MSE: {lr_mse_train:.2f}")
st.write(f"Test MSE: {lr_mse_test:.2f}")
st.write("**Huber Regression (Có Outliers):**")
# st.write(f"Intercept: {huber.intercept_:.2f}")
st.write(f"Train MSE: {huber_mse_train:.2f}")
st.write(f"Test MSE: {huber_mse_test:.2f}")

# Hiển thị biểu đồ 2D
st.subheader("Biểu đồ 2D: So sánh Linear và Huber Regression với dữ liệu tùy chỉnh")
st.write(
    "Linear Regression nhạy cảm với điểm ngoại lệ, còn Huber Regression thì "
    "mạnh mẽ hơn. Thêm điểm mới để xem sự khác biệt!"
)

lin_model, huber_model, lin_mse_2d, huber_mse_2d = fit_models(st.session_state.data)
st.plotly_chart(
    make_plot(st.session_state.data, lin_model, huber_model),
    use_container_width=True
)

# Hiển thị metrics mô hình 2D
st.write("**Linear Regression (2D):**")
# st.write(f"Intercept: {lin_model.intercept_:.2f}")
st.write(f"MSE: {lin_mse_2d:.2f}")
st.write("**Huber Regression (2D):**")
# st.write(f"Intercept: {huber_model.intercept_:.2f}")
st.write(f"MSE: {huber_mse_2d:.2f}")

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