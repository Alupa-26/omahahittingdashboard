import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Wedge, FancyBboxPatch
import matplotlib.colors as mcolors
import google.generativeai as genai
from io import BytesIO
import time

# ==========================================
# 1. SYSTEM CONFIGURATION & DESIGN SYSTEM
# ==========================================
st.set_page_config(page_title="Omaha Baseball Analytics", page_icon="⚾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        /* Typography & Spacing */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding: 2rem 3rem; max-width: 1600px; }
        
        /* Premium Header */
        .dashboard-header {
            background: linear-gradient(135deg, #0B0F19 0%, #1A2235 100%);
            border-radius: 16px; padding: 30px 40px; color: white; margin-bottom: 30px;
            border-left: 8px solid #D71920; /* Omaha Red */
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            display: flex; justify-content: space-between; align-items: center;
        }
        .header-title { font-size: 38px; font-weight: 900; margin: 0; letter-spacing: -1px; }
        .header-sub { font-size: 14px; font-weight: 600; color: #94A3B8; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 5px; }
        
        /* KPI Grid System */
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .kpi-card {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 15px; text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 20px -8px rgba(0,0,0,0.12); border-color: #D71920; }
        .kpi-label { font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
        .kpi-value { font-size: 26px; font-weight: 900; color: #0F172A; }
        .kpi-highlight { color: #D71920; } /* Brand accent for elite metrics */

        /* Tabs & Components */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #F8FAFC; padding: 10px 10px 0 10px; border-radius: 12px 12px 0 0; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 12px 24px; font-weight: 600; color: #64748B; border: none; }
        .stTabs [aria-selected="true"] { background-color: white !important; color: #0F172A !important; box-shadow: 0 -4px 6px -1px rgba(0,0,0,0.05); }
        
        /* Custom Dataframe */
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ARCHITECTURE & CACHING
# ==========================================
DB_NAME = "trackman_master.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trackman_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            PitchNo INTEGER, Date TEXT, Time TEXT, Pitcher TEXT, Batter TEXT, BatterSide TEXT,
            ExitSpeed REAL, Angle REAL, Direction REAL, Distance REAL, HitSpinRate REAL,
            HangTime REAL, PlateLocHeight REAL, PlateLocSide REAL,
            ContactPositionX REAL, ContactPositionY REAL, ContactPositionZ REAL,
            PitchCall TEXT, PlayResult TEXT, TaggedHitType TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_data_to_db(df):
    conn = sqlite3.connect(DB_NAME)
    existing_dates = pd.read_sql("SELECT DISTINCT Date FROM trackman_data", conn)['Date'].tolist()
    if not df.empty and df['Date'].iloc[0] in existing_dates:
        return False, "Data for this session date already exists in the database."
    df.to_sql('trackman_data', conn, if_exists='append', index=False)
    conn.close()
    fetch_data.clear() # Invalidate cache on new upload
    return True, "Data successfully ingested into master database."

@st.cache_data(ttl=3600)
def fetch_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM trackman_data", conn)
    conn.close()
    return df

init_db()

# ==========================================
# 3. ADVANCED ANALYTICS ENGINE
# ==========================================
def get_batter_side(side_str):
    if pd.isna(side_str): return "UNK"
    return 'LHH' if side_str.strip().lower() == 'left' else 'RHH' if side_str.strip().lower() == 'right' else 'SHH'

def assign_sz_zone(x, z):
    if pd.isna(x) or pd.isna(z): return None
    if -0.833 <= x <= 0.833 and 1.5 <= z <= 3.5:
        if z >= 2.833: return 'Z1' if x <= -0.277 else 'Z2' if x <= 0.277 else 'Z3'
        elif z >= 2.166: return 'Z4' if x <= -0.277 else 'Z5' if x <= 0.277 else 'Z6'
        else: return 'Z7' if x <= -0.277 else 'Z8' if x <= 0.277 else 'Z9'
    elif -1.166 <= x <= 1.166 and 3.5 < z <= 3.833: return 'S1'
    elif -1.166 <= x <= 1.166 and 1.166 <= z < 1.5: return 'S3'
    elif -1.166 <= x < -0.833 and 1.5 <= z <= 3.5: return 'S2'
    elif 0.833 < x <= 1.166 and 1.5 <= z <= 3.5: return 'S4'
    return None

def assign_field_zone(dist, direction):
    if pd.isna(dist) or pd.isna(direction): return None
    if dist < 180: 
        if direction < -22.5: return 'IF_LL'
        elif direction < 0: return 'IF_LC'
        elif direction < 22.5: return 'IF_RC'
        else: return 'IF_RR'
    else:          
        if direction < -15: return 'OF_L'
        elif direction < 15: return 'OF_C'
        else: return 'OF_R'

# ==========================================
# 4. PLOTLY 3D RENDERING ENGINE
# ==========================================
def render_3d_stadium(df):
    fig = go.Figure()
    
    # Lighting & Material Settings
    grass_material = dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1)
    dirt_material = dict(ambient=0.7, diffuse=0.7, roughness=1.0, specular=0.0)

    # Outfield Grass Striping
    theta_arc = np.linspace(-np.pi/4, np.pi/4, 150)
    for r in range(150, 401, 25):
        x_out, y_out = r * np.sin(theta_arc), r * np.cos(theta_arc)
        x_in, y_in = (r-25) * np.sin(theta_arc[::-1]), (r-25) * np.cos(theta_arc[::-1])
        stripe_color = '#2E7D32' if (r//25)%2 == 0 else '#388E3C'
        fig.add_trace(go.Mesh3d(
            x=np.concatenate([x_out, x_in, [x_out[0]]]), y=np.concatenate([y_out, y_in, [y_out[0]]]), z=np.zeros(301),
            color=stripe_color, lighting=grass_material, hoverinfo='skip', showlegend=False
        ))
        
    # Infield Dirt & Home Plate Circle
    theta_full = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Mesh3d(
        x=np.concatenate([[0], 145 * np.sin(np.linspace(-np.pi/4, np.pi/4, 50)), [0]]),
        y=np.concatenate([[0], 145 * np.cos(np.linspace(-np.pi/4, np.pi/4, 50)), [0]]),
        z=np.full(52, 0.1), color='#A1887F', lighting=dirt_material, hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Mesh3d(x=13*np.sin(theta_full), y=13*np.cos(theta_full), z=np.full(50, 0.15), color='#A1887F', lighting=dirt_material, hoverinfo='skip', showlegend=False)) # Home Plate Dirt
    fig.add_trace(go.Mesh3d(x=9*np.sin(theta_full), y=60.5 + 9*np.cos(theta_full), z=np.full(50, 0.15), color='#A1887F', lighting=dirt_material, hoverinfo='skip', showlegend=False)) # Pitcher Mound

    # Warning Track & Wall
    fig.add_trace(go.Mesh3d(
        x=np.concatenate([400 * np.sin(theta_arc), 385 * np.sin(theta_arc[::-1]), [400 * np.sin(theta_arc[0])]]),
        y=np.concatenate([400 * np.cos(theta_arc), 385 * np.cos(theta_arc[::-1]), [400 * np.cos(theta_arc[0])]]),
        z=np.full(301, 0.2), color='#8D6E63', lighting=dirt_material, hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(x=400 * np.sin(theta_arc), y=400 * np.cos(theta_arc), z=np.full(150, 10), mode='lines', surfaceaxis=2, surfacecolor='#0F172A', line=dict(width=5, color='#D71920'), hoverinfo='skip', showlegend=False)) # Padded Wall with Omaha Red top line

    # Bases & Lines
    fig.add_trace(go.Scatter3d(x=[63.6, 0, -63.6], y=[63.6, 127.2, 63.6], z=[0.3, 0.3, 0.3], mode='markers', marker=dict(color='white', size=5, symbol='square'), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 283], y=[0, 283], z=[0.2, 0.2], mode='lines', line=dict(color='white', width=3), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, -283], y=[0, 283], z=[0.2, 0.2], mode='lines', line=dict(color='white', width=3), hoverinfo='skip', showlegend=False))

    # Ball Trajectories
    for _, row in df.iterrows():
        dist, direction, angle, ev = row['Distance'], row['Direction'], row['Angle'], row['ExitSpeed']
        t = np.linspace(0, 1, 60) # High-res interpolation
        x_end, y_end = dist * np.sin(np.radians(direction)), dist * np.cos(np.radians(direction))
        x_traj, y_traj = t * x_end, t * y_end
        h_max = dist * np.tan(np.radians(max(0.1, angle))) * 0.25 
        z_traj = 4 * h_max * t * (1 - t)
        
        # Color coding: Barrel/Hard Hit -> Red, Solid -> Blue, Weak -> Gray
        color = '#D71920' if (ev >= 95 and 8 <= angle <= 32) else '#3B82F6' if ev >= 90 else '#94A3B8'
        
        fig.add_trace(go.Scatter3d(
            x=x_traj, y=y_traj, z=z_traj, mode='lines', line=dict(color=color, width=5),
            hovertemplate=f"<b>EV:</b> {ev:.1f} mph<br><b>LA:</b> {angle:.1f}°<br><b>Dist:</b> {dist:.0f} ft<extra></extra>", showlegend=False
        ))
        # Ground shadow tracking
        fig.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=np.full(60, 0.3), mode='lines', line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dot'), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x_end], y=[y_end], z=[0.3], mode='markers', marker=dict(color=color, size=6, line=dict(color='white', width=1)), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-300, 300], visible=False), yaxis=dict(range=[-20, 450], visible=False), zaxis=dict(range=[0, 150], visible=False),
            aspectmode='manual', aspectratio=dict(x=1.3, y=1.3, z=0.35),
            camera=dict(
                up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=-1.8, z=0.6) # Perfect catcher view
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=800, paper_bgcolor='rgba(0,0,0,0)',
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=0.05, xanchor="center", yanchor="bottom",
            buttons=list([
                dict(args=[{"scene.camera.eye": {"x": 0, "y": -1.8, "z": 0.6}}], label="Catcher View", method="relayout"),
                dict(args=[{"scene.camera.eye": {"x": 0, "y": 0.1, "z": 2.5}}], label="Top-Down", method="relayout"),
                dict(args=[{"scene.camera.eye": {"x": 1.5, "y": -1.5, "z": 1.0}}], label="Isometric", method="relayout")
            ])
        )]
    )
    return fig

def render_3d_strikezone(df):
    fig = go.Figure()
    
    # High-Fidelity Home Plate
    plate_x = [-0.708, 0.708, 0.708, 0, -0.708, -0.708]
    plate_y = [0, 0, 0.708, 1.417, 0.708, 0]
    fig.add_trace(go.Scatter3d(x=plate_x, y=plate_y, z=np.zeros(6), mode='lines', surfaceaxis=2, surfacecolor='white', line=dict(color='#0F172A', width=6), hoverinfo='skip', showlegend=False))
    
    # Volumetric Glass Strike Zone Box with 9-Zone Grid
    sz_z_lines = [1.5, 2.166, 2.833, 3.5]
    sz_x_lines = [-0.833, -0.277, 0.277, 0.833]
    
    # Front and Back Faces
    for y_plane in [0, 1.417]:
        fig.add_trace(go.Scatter3d(x=[-0.833, 0.833, 0.833, -0.833, -0.833], y=np.full(5, y_plane), z=[1.5, 1.5, 3.5, 3.5, 1.5], mode='lines', line=dict(color='rgba(215, 25, 32, 0.8)' if y_plane==0 else 'rgba(148, 163, 184, 0.4)', width=4), hoverinfo='skip', showlegend=False))
        # Inner Grid Lines
        for z in sz_z_lines[1:3]: fig.add_trace(go.Scatter3d(x=[-0.833, 0.833], y=[y_plane, y_plane], z=[z, z], mode='lines', line=dict(color='rgba(148, 163, 184, 0.3)', width=2), hoverinfo='skip', showlegend=False))
        for x in sz_x_lines[1:3]: fig.add_trace(go.Scatter3d(x=[x, x], y=[y_plane, y_plane], z=[1.5, 3.5], mode='lines', line=dict(color='rgba(148, 163, 184, 0.3)', width=2), hoverinfo='skip', showlegend=False))

    # Corner Connectors
    for x in [-0.833, 0.833]:
        for z in [1.5, 3.5]:
            fig.add_trace(go.Scatter3d(x=[x, x], y=[0, 1.417], z=[z, z], mode='lines', line=dict(color='rgba(148, 163, 184, 0.5)', width=3), hoverinfo='skip', showlegend=False))

    # Plotted Contact Points
    depth_col = 'ContactPositionZ' if 'ContactPositionZ' in df.columns else 'ContactPositionY'
    valid_cp = df.dropna(subset=['PlateLocSide', 'PlateLocHeight', depth_col])
    
    if not valid_cp.empty:
        fig.add_trace(go.Scatter3d(
            x=valid_cp['PlateLocSide'], y=valid_cp[depth_col], z=valid_cp['PlateLocHeight'], mode='markers',
            marker=dict(size=9, color=valid_cp['ExitSpeed'], colorscale='Turbo', showscale=True, colorbar=dict(title="EV (mph)", len=0.75, thickness=15, outlinewidth=0, tickfont=dict(family="Inter")), line=dict(color='black', width=1)),
            hovertemplate="<b style='color:#0F172A'>EV: %{marker.color:.1f} mph</b><br>Side (X): %{x:.2f} ft<br>Depth (Y): %{y:.2f} ft<br>Height (Z): %{z:.2f} ft<extra></extra>",
            name='Contact'
        ))
        # Drop shadows on the plate
        fig.add_trace(go.Scatter3d(x=valid_cp['PlateLocSide'], y=valid_cp[depth_col], z=np.zeros(len(valid_cp)), mode='markers', marker=dict(color='rgba(0,0,0,0.2)', size=5), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Side (ft)', range=[-2.5, 2.5], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True),
            yaxis=dict(title='Depth (ft)', range=[-1, 3], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True),
            zaxis=dict(title='Height (ft)', range=[0, 4.5], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True),
            aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.9),
            camera=dict(eye=dict(x=1.2, y=-1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=700, paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 5. MATPLOTLIB PDF EXPORT ENGINE
# ==========================================
# (Retained constants and precise Matplotlib drawing functions from previous iteration for PDF generation)
ZONES_CONFIG = { 'IF_LL': {'r_inner': 0, 'width': 180, 'theta1': 112.5, 'theta2': 135, 'text_r': 160, 'text_theta': 123.75}, 'IF_LC': {'r_inner': 0, 'width': 180, 'theta1': 90, 'theta2': 112.5, 'text_r': 160, 'text_theta': 101.25}, 'IF_RC': {'r_inner': 0, 'width': 180, 'theta1': 67.5, 'theta2': 90, 'text_r': 160, 'text_theta': 78.75}, 'IF_RR': {'r_inner': 0, 'width': 180, 'theta1': 45, 'theta2': 67.5, 'text_r': 160, 'text_theta': 56.25}, 'OF_L': {'r_inner': 180, 'width': 200, 'theta1': 105, 'theta2': 135, 'text_r': 350, 'text_theta': 120}, 'OF_C': {'r_inner': 180, 'width': 200, 'theta1': 75, 'theta2': 105, 'text_r': 350, 'text_theta': 90}, 'OF_R': {'r_inner': 180, 'width': 200, 'theta1': 45, 'theta2': 75, 'text_r': 350, 'text_theta': 60} }
SZ_ZONES_CONFIG = { 'Z1': {'x': -0.833, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z2': {'x': -0.277, 'y': 2.833, 'w': 0.554, 'h': 0.667}, 'Z3': {'x': 0.277, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z4': {'x': -0.833, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z5': {'x': -0.277, 'y': 2.166, 'w': 0.554, 'h': 0.667}, 'Z6': {'x': 0.277, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z7': {'x': -0.833, 'y': 1.5, 'w': 0.556, 'h': 0.666}, 'Z8': {'x': -0.277, 'y': 1.5, 'w': 0.554, 'h': 0.666}, 'Z9': {'x': 0.277, 'y': 1.5, 'w': 0.556, 'h': 0.666}, 'S1': {'x': -1.166, 'y': 3.5, 'w': 2.332, 'h': 0.333}, 'S3': {'x': -1.166, 'y': 1.167, 'w': 2.332, 'h': 0.333}, 'S2': {'x': -1.166, 'y': 1.5, 'w': 0.333, 'h': 2.0}, 'S4': {'x': 0.833, 'y': 1.5, 'w': 0.333, 'h': 2.0} }

def generate_pdf_buffer(df, kpis, player_name, side, date_str):
    fig = plt.figure(figsize=(14, 11)) 
    fig.patches.append(Rectangle((0.015, 0.015), 0.97, 0.97, fill=False, edgecolor='#0F172A', lw=2, transform=fig.transFigure))
    
    fig.text(0.04, 0.94, f"{player_name.upper()}", fontsize=28, fontweight='900', color='#0F172A')
    fig.text(0.35, 0.94, f"|  {side}", fontsize=26, fontweight='300', color='#D71920')
    fig.text(0.04, 0.91, f"OMAHA EXECUTIVE ANALYTICS • SESSION: {date_str}", fontsize=12, color='#64748B', fontweight='bold', letter_spacing=1)
    fig.add_artist(plt.Line2D((0.04, 0.89), (0.89, 0.89), color='#E2E8F0', linewidth=2))

    box_width = 0.085 
    x_offsets = np.linspace(0.025, 0.885, 9).tolist()
    
    for idx, (label, val) in enumerate(kpis):
        fig.patches.append(FancyBboxPatch((x_offsets[idx], 0.80), box_width, 0.065, boxstyle="round,pad=0.01", fc="#F8FAFC", ec="#CBD5E1", lw=1.5, transform=fig.transFigure))
        cx = x_offsets[idx] + (box_width / 2)
        fig.text(cx, 0.845, label, fontsize=8, fontweight='bold', color='#64748B', ha='center')
        fig.text(cx, 0.815, str(val), fontsize=14, fontweight='900', color='#0F172A', ha='center')

    def _draw_field(ax):
        angles = np.linspace(-45, 45, 100)
        ax.plot(380 * np.sin(np.radians(angles)), 380 * np.cos(np.radians(angles)), color='#475569', lw=2, zorder=3)
        ax.add_patch(Wedge(center=(0, 0), r=380, theta1=45, theta2=135, facecolor='#F8FAFC', edgecolor='black', lw=1, zorder=1))
        ax.add_patch(Wedge(center=(0, 0), r=154, theta1=45, theta2=135, facecolor='#F1F5F9', edgecolor='black', lw=1, zorder=2))
        for a in [-22.5, 0, 22.5]: ax.plot([0, 380 * np.sin(np.radians(a))], [0, 380 * np.cos(np.radians(a))], color='#CBD5E1', lw=1, zorder=3)
        ax.plot([0, 380 * np.sin(np.radians(-45))], [0, 380 * np.cos(np.radians(-45))], color='#475569', lw=2, zorder=3)
        ax.plot([0, 380 * np.sin(np.radians(45))], [0, 380 * np.cos(np.radians(45))], color='#475569', lw=2, zorder=3)

    # 1. Spray Chart
    ax1 = fig.add_axes([0.025, 0.42, 0.28, 0.34])
    _draw_field(ax1)
    ax1.scatter(df['Distance'] * np.sin(np.radians(df['Direction'])), df['Distance'] * np.cos(np.radians(df['Direction'])), c=df['ExitSpeed'], cmap='coolwarm', vmin=75, vmax=105, s=40, edgecolors='black', lw=0.5, zorder=6)
    ax1.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Batted Ball Spray (Color = EV)")

    # 2. EV by Quadrant
    ax2 = fig.add_axes([0.345, 0.42, 0.28, 0.34])
    _draw_field(ax2)
    df_copy = df.copy()
    df_copy['Zone'] = df_copy.apply(lambda r: assign_field_zone(r['Distance'], r['Direction']), axis=1)
    avgs = df_copy.groupby('Zone')['ExitSpeed'].mean()
    norm, cmap = mcolors.Normalize(vmin=80, vmax=100), plt.get_cmap('coolwarm')
    for z, c in ZONES_CONFIG.items():
        val = avgs.get(z, np.nan)
        ax2.add_patch(Wedge(center=(0, 0), r=c['r_inner']+c['width'], theta1=c['theta1'], theta2=c['theta2'], width=c['width'], facecolor=cmap(norm(val)) if not pd.isna(val) else '#FFFFFF', edgecolor='#CBD5E1', alpha=0.9, zorder=1))
        if not pd.isna(val): ax2.text(c['text_r']*np.cos(np.radians(c['text_theta'])), c['text_r']*np.sin(np.radians(c['text_theta'])), f"{val:.1f}", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2), zorder=4)
    ax2.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Avg EV by Quadrant")

    # 3. LA by Quadrant
    ax3 = fig.add_axes([0.665, 0.42, 0.28, 0.34])
    _draw_field(ax3)
    avgs_la = df_copy.groupby('Zone')['Angle'].mean()
    norm_la, cmap_la = mcolors.Normalize(vmin=0, vmax=35), plt.get_cmap('viridis')
    for z, c in ZONES_CONFIG.items():
        val = avgs_la.get(z, np.nan)
        ax3.add_patch(Wedge(center=(0, 0), r=c['r_inner']+c['width'], theta1=c['theta1'], theta2=c['theta2'], width=c['width'], facecolor=cmap_la(norm_la(val)) if not pd.isna(val) else '#FFFFFF', edgecolor='#CBD5E1', alpha=0.9, zorder=1))
        if not pd.isna(val): ax3.text(c['text_r']*np.cos(np.radians(c['text_theta'])), c['text_r']*np.sin(np.radians(c['text_theta'])), f"{val:.1f}°", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2), zorder=4)
    ax3.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Avg LA by Quadrant")

    def _draw_sz(ax, metric, vmin, vmax, cmap_name, title, is_ev=True):
        df_copy['SZ'] = df_copy.apply(lambda r: assign_sz_zone(r['PlateLocSide'], r['PlateLocHeight']), axis=1)
        z_avgs = df_copy.groupby('SZ')[metric].mean()
        n, cm = mcolors.Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap_name)
        for z, r in SZ_ZONES_CONFIG.items():
            v = z_avgs.get(z, np.nan)
            ax.add_patch(Rectangle((r['x'], r['y']), r['w'], r['h'], facecolor=cm(n(v)) if not pd.isna(v) else '#F8FAFC', edgecolor='#94A3B8', zorder=1))
            if not pd.isna(v): ax.text(r['x']+r['w']/2, r['y']+r['h']/2, f"{v:.1f}", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=3)
        ax.add_patch(Rectangle((-0.833, 1.5), 1.666, 2.0, fill=False, edgecolor='black', lw=2.5, zorder=2))
        ax.add_patch(MplPolygon(np.column_stack(([-0.708, 0.708, 0.708, 0, -0.708], [0, 0, 0.25, 0.5, 0.25])), facecolor='white', edgecolor='black', lw=1.5, zorder=2))
        ax.set(xlim=(-2, 2), ylim=(0, 4.5), aspect='equal', xticks=[], yticks=[], title=title)

    # 4 & 5. Strike Zone Plots
    ax4 = fig.add_axes([0.025, 0.06, 0.28, 0.34])
    _draw_sz(ax4, 'ExitSpeed', 80, 100, 'coolwarm', "Avg EV by Pitch Location")
    ax5 = fig.add_axes([0.345, 0.06, 0.28, 0.34])
    _draw_sz(ax5, 'Angle', 0, 35, 'viridis', "Avg LA by Pitch Location", is_ev=False)

    # 6. Contact Depth
    ax6 = fig.add_axes([0.665, 0.06, 0.28, 0.34])
    dep_col = 'ContactPositionZ' if 'ContactPositionZ' in df.columns else 'ContactPositionY'
    cp_x = df['ContactPositionX'] * 12 if 'ContactPositionX' in df.columns else df['PlateLocSide'] * 12
    cp_z = df[dep_col] * 12 if dep_col in df.columns else np.zeros(len(df))
    ax6.grid(True, linestyle='--', alpha=0.6, color='#CBD5E1', zorder=0)
    ax6.axhline(0, color='black', lw=1.5, zorder=1); ax6.axvline(0, color='black', lw=1.5, zorder=1)
    ax6.add_patch(MplPolygon(np.column_stack(([0, -8.5, -8.5, 8.5, 8.5, 0], [0, 8.5, 17, 17, 8.5, 0])), facecolor='#F1F5F9', edgecolor='black', lw=1.5, zorder=2))
    ax6.scatter(cp_z, cp_x, c=df['ExitSpeed'], cmap='coolwarm', vmin=75, vmax=105, s=50, edgecolors='black', lw=0.5, zorder=5)
    ax6.set(xlim=(-30, 30), ylim=(-5, 55), aspect='equal', title="Contact Depth (inches)", xlabel="Depth Out In Front", ylabel="Side to Side")

    buf = BytesIO()
    fig.savefig(buf, format='pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# ==========================================
# 6. APPLICATION STATE & SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Omaha_Mavericks_logo.svg/1200px-Omaha_Mavericks_logo.svg.png", width=140)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📥 Database Ingestion")
    uploaded_file = st.file_uploader("Upload Trackman CSV", type=['csv'], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding='latin1')
            if st.button("Process & Ingest Data", type="primary", use_container_width=True):
                with st.spinner("Writing to Master SQLite Database..."):
                    success, msg = load_data_to_db(df_upload)
                    if success: st.toast("✅ " + msg)
                    else: st.warning(msg)
        except Exception as e:
            st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Session Filters")
    all_data = fetch_data()
    
    if not all_data.empty:
        batters = sorted(all_data['Batter'].dropna().unique())
        selected_batter = st.selectbox("Hitter Profile", batters)
        
        batter_subset = all_data[all_data['Batter'] == selected_batter]
        dates = ["All-Time"] + sorted(batter_subset['Date'].dropna().unique().tolist(), reverse=True)
        selected_date = st.selectbox("Date Range", dates)
        
        if selected_date != "All-Time":
            batter_subset = batter_subset[batter_subset['Date'] == selected_date]
            
        batted_balls = batter_subset.dropna(subset=['ExitSpeed', 'Angle', 'Direction', 'Distance'])
    else:
        st.info("System Offline. Ingest Trackman CSV to initialize engines.")
        batted_balls = pd.DataFrame()
        selected_batter = None

    st.markdown("---")
    st.markdown("### 🧠 API Configuration")
    gemini_key = st.text_input("Gemini Pro API Key", type="password", placeholder="Enter key for AI generation...")

# ==========================================
# 7. MAIN EXECUTIVE DASHBOARD
# ==========================================
if not batted_balls.empty and selected_batter:
    
    # --- Sabermetric Computations ---
    total_swings = len(batter_subset[batter_subset['PitchCall'].isin(['InPlay', 'Foul', 'StrikeSwinging'])])
    bip = len(batted_balls)
    avg_ev, max_ev = batted_balls['ExitSpeed'].mean(), batted_balls['ExitSpeed'].max()
    ev90 = np.percentile(batted_balls['ExitSpeed'], 90) # Elite indicator
    
    hard_hits = len(batted_balls[batted_balls['ExitSpeed'] >= 95]) # MLB threshold scaling
    hh_pct = (hard_hits / bip) * 100 if bip > 0 else 0
    
    sweet_spots = len(batted_balls[(batted_balls['Angle'] >= 8) & (batted_balls['Angle'] <= 32)])
    swsp_pct = (sweet_spots / bip) * 100 if bip > 0 else 0
    
    avg_la, std_la = batted_balls['Angle'].mean(), batted_balls['Angle'].std()
    avg_dist = batted_balls['Distance'].mean()
    side = get_batter_side(batter_subset['BatterSide'].iloc[0] if 'BatterSide' in batter_subset.columns else "UNK")

    # --- Header Render ---
    st.markdown(f"""
        <div class="dashboard-header">
            <div>
                <div class="header-title">{selected_batter.upper()}</div>
                <div class="header-sub">BATS: {side} &nbsp;|&nbsp; RANGE: {selected_date} &nbsp;|&nbsp; BATTED BALL EVENTS: {bip}</div>
            </div>
            <img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Omaha_Mavericks_logo.svg/1200px-Omaha_Mavericks_logo.svg.png" style="height:60px; opacity:0.9;">
        </div>
    """, unsafe_allow_html=True)
    
    # --- KPI Grid Render ---
    pdf_kpis = [
        ("TOTAL SWINGS", total_swings), ("HARD HIT %", f"{hh_pct:.1f}%"), ("SWEET SPOT %", f"{swsp_pct:.1f}%"),
        ("AVG EV", f"{avg_ev:.1f}"), ("90th% EV", f"{ev90:.1f}"), ("MAX EV", f"{max_ev:.1f}"),
        ("AVG LA", f"{avg_la:.1f}°"), ("LA DEV (SD)", f"{std_la:.1f}°"), ("AVG DIST", f"{avg_dist:.0f} ft")
    ]
    
    grid_html = '<div class="kpi-grid">'
    for label, val in pdf_kpis:
        val_class = "kpi-value kpi-highlight" if label in ["90th% EV", "SWEET SPOT %", "HARD HIT %"] else "kpi-value"
        grid_html += f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="{val_class}">{val}</div></div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # --- Engine Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Field Trajectory Engine", "⚾ Volumetric Strike Zone", "🧠 LLM Scouting Synthesis", "📄 Executive Reporting"])
    
    with tab1:
        st.plotly_chart(render_3d_stadium(batted_balls), use_container_width=True)

    with tab2:
        st.plotly_chart(render_3d_strikezone(batted_balls), use_container_width=True)

    with tab3:
        st.markdown("### Automated Hitter Profile Generation")
        if gemini_key:
            if st.button("Synthesize Report via Gemini Pro", type="primary"):
                with st.spinner("Running generative sabermetric analysis..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        df_ai = batted_balls.copy()
                        df_ai['SZ_Zone'] = df_ai.apply(lambda r: assign_sz_zone(r['PlateLocSide'], r['PlateLocHeight']), axis=1)
                        zone_evs = df_ai.groupby('SZ_Zone')['ExitSpeed'].mean().round(1).to_dict()
                        
                        prompt = f"""
                        You are the Director of Player Development. Write an elite, highly professional scouting report based on Trackman data for {selected_batter} ({side}).
                        
                        SESSION DATA: 
                        Swings: {total_swings}, Batted Balls: {bip}
                        Avg EV: {avg_ev:.1f}, 90th% EV: {ev90:.1f}, Max EV: {max_ev:.1f}
                        Sweet Spot % (8-32 deg): {swsp_pct:.1f}%
                        Avg LA: {avg_la:.1f}° (StdDev: {std_la:.1f}°)
                        Zone EVs (Z1-Z3 High, Z7-Z9 Low): {zone_evs}
                        
                        FORMAT AS MARKDOWN. Do not use generic filler. Be highly analytical.
                        - **Section 1: Power & Batted Ball Profile** (Analyze EV90 vs Max, Sweet Spot rate, and LA standard deviation for consistency).
                        - **Section 2: Zone Optimization** (Where do they do damage? Where are the holes based on the Zone EVs?)
                        - **Section 3: Development Directive** (One clear, actionable mechanical or approach adjustment for the cage).
                        """
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"API Error: {e}")
        else:
            st.info("Unlock AI Generation by providing a Gemini API Key in the configuration sidebar.")

    with tab4:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Export Capabilities")
            st.write("Generate a high-resolution, 6-chart vector PDF report designed for print or iPad review by the coaching staff. Compilation requires ~3 seconds.")
            
            if st.button("Generate Executive PDF", type="primary"):
                with st.spinner("Compiling Matplotlib vector graphics..."):
                    pdf_bytes = generate_pdf_buffer(batted_balls, pdf_kpis, selected_batter, side, selected_date)
                    st.download_button(
                        label="⬇️ Download Document",
                        data=pdf_bytes,
                        file_name=f"Omaha_Dev_Report_{selected_batter.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
        
        with col2:
            st.markdown("### Database Architecture View")
            display_cols = ['PitchNo', 'Pitcher', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'HitSpinRate']
            st.dataframe(batted_balls[display_cols].sort_values('ExitSpeed', ascending=False).head(100), use_container_width=True, height=350)
