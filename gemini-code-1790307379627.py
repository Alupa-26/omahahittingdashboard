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

# --- PAGE SETUP & THEMING ---
st.set_page_config(page_title="Omaha Player Development Dashboard", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .kpi-card {
            background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 15px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 15px;
        }
        .kpi-label { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; }
        .kpi-value { font-size: 20px; font-weight: 800; color: #000000; }
        .header-box {
            background: linear-gradient(90deg, #111111 0%, #2b2b2b 100%);
            border-radius: 8px; padding: 20px; color: white; margin-bottom: 20px;
        }
        .header-title { font-size: 28px; font-weight: 800; margin: 0; }
        .header-sub { font-size: 14px; font-weight: 500; color: #cbd5e1; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- DATABASE MANAGEMENT ---
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
    return True, "Successfully loaded into database."

def fetch_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM trackman_data", conn)
    conn.close()
    return df

init_db()

# --- HELPER FUNCTIONS ---
def get_batter_side(side_str):
    if pd.isna(side_str): return "Unk"
    s = str(side_str).strip().lower()
    return 'LHH' if s == 'left' else 'RHH' if s == 'right' else 'SHH' if s == 'switch' else s

def assign_zone(dist, direction):
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

def assign_sz_zone(x, z):
    if pd.isna(x) or pd.isna(z): return None
    if -0.833 <= x <= 0.833 and 1.5 <= z <= 3.5:
        if z >= 2.833:
            return 'Z1' if x <= -0.277 else 'Z2' if x <= 0.277 else 'Z3'
        elif z >= 2.166:
            return 'Z4' if x <= -0.277 else 'Z5' if x <= 0.277 else 'Z6'
        else:
            return 'Z7' if x <= -0.277 else 'Z8' if x <= 0.277 else 'Z9'
    elif -1.166 <= x <= 1.166 and 3.5 < z <= 3.833: return 'S1'
    elif -1.166 <= x <= 1.166 and 1.166 <= z < 1.5: return 'S3'
    elif -1.166 <= x < -0.833 and 1.5 <= z <= 3.5: return 'S2'
    elif 0.833 < x <= 1.166 and 1.5 <= z <= 3.5: return 'S4'
    return None

# --- PDF MATPLOTLIB CONSTANTS & DRAWING FUNCTIONS ---
ZONES_CONFIG = {
    'IF_LL': {'r_inner': 0, 'width': 180, 'theta1': 112.5, 'theta2': 135, 'text_r': 160, 'text_theta': 123.75},
    'IF_LC': {'r_inner': 0, 'width': 180, 'theta1': 90, 'theta2': 112.5, 'text_r': 160, 'text_theta': 101.25},
    'IF_RC': {'r_inner': 0, 'width': 180, 'theta1': 67.5, 'theta2': 90, 'text_r': 160, 'text_theta': 78.75},
    'IF_RR': {'r_inner': 0, 'width': 180, 'theta1': 45, 'theta2': 67.5, 'text_r': 160, 'text_theta': 56.25},
    'OF_L':  {'r_inner': 180, 'width': 200, 'theta1': 105, 'theta2': 135, 'text_r': 350, 'text_theta': 120},
    'OF_C':  {'r_inner': 180, 'width': 200, 'theta1': 75, 'theta2': 105, 'text_r': 350, 'text_theta': 90},
    'OF_R':  {'r_inner': 180, 'width': 200, 'theta1': 45, 'theta2': 75, 'text_r': 350, 'text_theta': 60},
}
SZ_ZONES_CONFIG = {
    'Z1': {'x': -0.833, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z2': {'x': -0.277, 'y': 2.833, 'w': 0.554, 'h': 0.667}, 'Z3': {'x': 0.277,  'y': 2.833, 'w': 0.556, 'h': 0.667},
    'Z4': {'x': -0.833, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z5': {'x': -0.277, 'y': 2.166, 'w': 0.554, 'h': 0.667}, 'Z6': {'x': 0.277,  'y': 2.166, 'w': 0.556, 'h': 0.667},
    'Z7': {'x': -0.833, 'y': 1.5,   'w': 0.556, 'h': 0.666}, 'Z8': {'x': -0.277, 'y': 1.5,   'w': 0.554, 'h': 0.666}, 'Z9': {'x': 0.277,  'y': 1.5,   'w': 0.556, 'h': 0.666},
    'S1': {'x': -1.166, 'y': 3.5,   'w': 2.332, 'h': 0.333}, 'S3': {'x': -1.166, 'y': 1.167, 'w': 2.332, 'h': 0.333},
    'S2': {'x': -1.166, 'y': 1.5,   'w': 0.333, 'h': 2.0},   'S4': {'x': 0.833,  'y': 1.5,   'w': 0.333, 'h': 2.0},
}

def draw_field_skeleton(ax, draw_infield=False):
    angles = np.linspace(-45, 45, 100)
    ax.plot(380 * np.sin(np.radians(angles)), 380 * np.cos(np.radians(angles)), color='#475569', lw=2, zorder=3)
    if draw_infield:
        ax.add_patch(Wedge(center=(0, 0), r=380, theta1=45, theta2=135, facecolor='#e6f4ea', edgecolor='black', lw=2, zorder=1))
        ax.add_patch(Wedge(center=(0, 0), r=154, theta1=45, theta2=135, facecolor='#f3e9d2', edgecolor='black', lw=1.5, zorder=2))
        for angle in [-22.5, 0, 22.5]: ax.plot([0, 380 * np.sin(np.radians(angle))], [0, 380 * np.cos(np.radians(angle))], color='black', lw=1, zorder=3)
        bases_x, bases_y = [90*np.sin(np.radians(45)), 0, 90*np.sin(np.radians(-45))], [90*np.cos(np.radians(45)), 90*np.sqrt(2), 90*np.cos(np.radians(-45))]
        for bx, by in zip(bases_x, bases_y): ax.add_patch(MplPolygon([[bx, by+4], [bx+4, by], [bx, by-4], [bx-4, by]], facecolor='white', edgecolor='black', lw=1, zorder=4))
        ax.scatter(0, 0, color='red', s=45, zorder=5)
    else:
        ax.plot([0, 380 * np.sin(np.radians(-45))], [0, 380 * np.cos(np.radians(-45))], color='#475569', lw=2, zorder=3)
        ax.plot([0, 380 * np.sin(np.radians(45))], [0, 380 * np.cos(np.radians(45))], color='#475569', lw=2, zorder=3)
        ax.scatter(0, 0, marker='D', color='#94a3b8', s=40, zorder=4)

def draw_zoned_plot(ax, df, metric, title, vmin, vmax, cmap='coolwarm', unit='', is_ev=False):
    draw_field_skeleton(ax, draw_infield=False) 
    df_copy = df.copy()
    df_copy['Zone'] = df_copy.apply(lambda row: assign_zone(row['Distance'], row['Direction']), axis=1)
    zone_avgs = df_copy.groupby('Zone')[metric].mean()
    norm, colormap = mcolors.Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap)
    for zone_name, config in ZONES_CONFIG.items():
        avg_val = zone_avgs.get(zone_name, np.nan)
        color = colormap(norm(avg_val)) if not pd.isna(avg_val) else '#f1f5f9'
        text_val = f"{avg_val:.1f}{unit}" if not pd.isna(avg_val) else "N/A"
        font_size = 8 if is_ev and zone_name.startswith('IF') else 11
        ax.add_patch(Wedge(center=(0, 0), r=config['r_inner'] + config['width'], theta1=config['theta1'], theta2=config['theta2'], width=config['width'], facecolor=color, edgecolor='#cbd5e1', lw=1.5, alpha=0.85, zorder=1))
        text_x, text_y = config['text_r'] * np.cos(np.radians(config['text_theta'])), config['text_r'] * np.sin(np.radians(config['text_theta']))
        ax.text(text_x, text_y, text_val, ha='center', va='center', fontsize=font_size, fontweight='bold', color='#0f172a', bbox=dict(facecolor='white', alpha=0.75, edgecolor='#e2e8f0', boxstyle='round,pad=0.3'), zorder=4)
    ax.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title=title)

def draw_strike_zone_plot(ax, df, metric, title, vmin, vmax, cmap='coolwarm', unit='', is_ev=False):
    df_copy = df.copy()
    df_copy['SZ_Zone'] = df_copy.apply(lambda row: assign_sz_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
    zone_avgs = df_copy.groupby('SZ_Zone')[metric].mean()
    norm, colormap = mcolors.Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap)
    for zone, rect in SZ_ZONES_CONFIG.items():
        avg_val = zone_avgs.get(zone, np.nan)
        color = colormap(norm(avg_val)) if not pd.isna(avg_val) else '#f1f5f9'
        text_val = (f"{avg_val:.1f}{unit}" if not is_ev else f"{avg_val:.1f}") if not pd.isna(avg_val) else "N/A"
        ax.add_patch(Rectangle((rect['x'], rect['y']), rect['w'], rect['h'], facecolor=color, edgecolor='#94a3b8', lw=1.5, zorder=1))
        font_size = 10 if zone.startswith('Z') else 8
        ax.text(rect['x'] + rect['w']/2, rect['y'] + rect['h']/2, text_val, ha='center', va='center', fontsize=font_size, fontweight='bold', color='#0f172a', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=3)
    ax.add_patch(Rectangle((-0.833, 1.5), 1.666, 2.0, fill=False, edgecolor='black', lw=2.5, zorder=2))
    ax.add_patch(MplPolygon(np.column_stack(([-0.708, 0.708, 0.708, 0, -0.708], [0, 0, 0.25, 0.5, 0.25])), facecolor='white', edgecolor='black', lw=1.5, zorder=2))
    ax.set(xlim=(-2, 2), ylim=(0, 4.5), aspect='equal', xticks=[], yticks=[], title=title)

def draw_contact_plot(ax, df, title="Contact Depth Position"):
    df_cp = df.copy()
    # Resolve depth proxy if needed based on Trackman schema
    df_cp['CP_X_in'] = df_cp['ContactPositionX'] * 12 if 'ContactPositionX' in df_cp.columns else df_cp['PlateLocSide'] * 12
    depth_col = 'ContactPositionZ' if 'ContactPositionZ' in df_cp.columns else 'ContactPositionY'
    df_cp['CP_Z_in'] = df_cp[depth_col] * 12 if depth_col in df_cp.columns else 0
    
    ax.grid(True, linestyle='--', alpha=0.6, color='#cbd5e1', zorder=0)
    ax.axhline(0, color='black', lw=1.5, zorder=1)
    ax.axvline(0, color='black', lw=1.5, zorder=1)
    ax.add_patch(MplPolygon(np.column_stack(([0, -8.5, -8.5, 8.5, 8.5, 0], [0, 8.5, 17, 17, 8.5, 0])), facecolor='#f1f5f9', edgecolor='black', lw=1.5, zorder=2))
    sc = ax.scatter(df_cp['CP_Z_in'], df_cp['CP_X_in'], c=df_cp['ExitSpeed'], cmap='coolwarm', vmin=75, vmax=100, s=65, edgecolors='#0f172a', lw=0.75, zorder=5)
    ax.set(xlim=(-30, 30), ylim=(-5, 55), aspect='equal', title=title, xlabel="Side to Side (inches)", ylabel="Out in Front (inches)")

# --- SIDEBAR & FILTERING ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Omaha_Mavericks_logo.svg/1200px-Omaha_Mavericks_logo.svg.png", width=120)
    st.markdown("### Data Management")
    uploaded_file = st.file_uploader("Upload Trackman CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding='latin1')
            cols_to_keep = ['PitchNo', 'Date', 'Time', 'Pitcher', 'Batter', 'BatterSide', 
                            'ExitSpeed', 'Angle', 'Direction', 'Distance', 'HitSpinRate', 
                            'HangTime', 'PlateLocHeight', 'PlateLocSide', 'ContactPositionX', 
                            'ContactPositionY', 'ContactPositionZ', 'PitchCall', 'PlayResult', 'TaggedHitType']
            available_cols = [c for c in cols_to_keep if c in df_upload.columns]
            df_cleaned = df_upload[available_cols]
            
            if st.button("Save to Database"):
                success, msg = load_data_to_db(df_cleaned)
                if success: st.success(msg)
                else: st.warning(msg)
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("### API Keys")
    gemini_key = st.text_input("Gemini API Key (For AI Analysis)", type="password")

    st.markdown("---")
    st.markdown("### Dashboard Filters")
    all_data = fetch_data()
    
    if not all_data.empty:
        batters = sorted(all_data['Batter'].dropna().unique())
        selected_batter = st.selectbox("Select Hitter:", batters)
        
        batter_subset = all_data[all_data['Batter'] == selected_batter]
        dates = ["All-Time"] + sorted(batter_subset['Date'].dropna().unique().tolist(), reverse=True)
        selected_date = st.selectbox("Select Session:", dates)
        
        if selected_date != "All-Time":
            batter_subset = batter_subset[batter_subset['Date'] == selected_date]
            
        batted_balls = batter_subset.dropna(subset=['ExitSpeed', 'Angle', 'Direction', 'Distance'])
    else:
        st.warning("Database empty. Please upload a CSV.")
        batted_balls = pd.DataFrame()
        selected_batter = None

# --- MAIN DASHBOARD ---
if not batted_balls.empty and selected_batter:
    # Key Metrics
    total_swings = len(batter_subset[batter_subset['PitchCall'].isin(['InPlay', 'Foul', 'StrikeSwinging'])])
    total_bip = len(batted_balls)
    max_ev = batted_balls['ExitSpeed'].max()
    hh_threshold = max_ev * 0.90
    total_hh = len(batted_balls[batted_balls['ExitSpeed'] >= hh_threshold])
    hh_pct = (total_hh / total_bip) * 100 if total_bip > 0 else 0
    avg_ev = batted_balls['ExitSpeed'].mean()
    avg_la = batted_balls['Angle'].mean()
    avg_dist = batted_balls['Distance'].mean()
    max_dist = batted_balls['Distance'].max()
    avg_spin = batted_balls['HitSpinRate'].mean()
    avg_hang = batted_balls['HangTime'].mean()
    
    batter_side = get_batter_side(batter_subset['BatterSide'].iloc[0] if 'BatterSide' in batter_subset.columns else "Unk")
    
    st.markdown(f"""
        <div class="header-box">
            <div class="header-title">{selected_batter} <span style="color:#e02424; font-size:22px;">| {batter_side}</span></div>
            <div class="header-sub">OMAHA PLAYER DEVELOPMENT • Session: {selected_date}</div>
        </div>
    """, unsafe_allow_html=True)
    
    kpi_cols = st.columns(10)
    kpis = [
        ("Swings", int(total_swings)), ("Total HH", int(total_hh)), ("HH %", f"{hh_pct:.1f}%"), 
        ("Avg EV", f"{avg_ev:.1f}"), ("Max EV", f"{max_ev:.1f}"), ("Avg LA", f"{avg_la:.1f}°"), 
        ("Avg Dist", f"{avg_dist:.0f}"), ("Max Dist", f"{max_dist:.0f}"), ("Avg Spin", f"{avg_spin:.0f}"), ("Hang", f"{avg_hang:.2f}s")
    ]
    for col, (label, val) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Field & Trajectories", "⚾ 3D Contact & Strike Zone", "🧠 AI Scouting Analysis", "📄 Professional PDF Export"])
    
    with tab1:
        st.markdown("### Interactive 3D Batted Ball Visualizer")
        fig_3d = go.Figure()
        
        # 3D FIELD RENDER (Baseball Look)
        # Outfield Grass (Mesh)
        theta = np.linspace(-np.pi/4, np.pi/4, 50)
        x_grass = np.concatenate(([0], 400 * np.sin(theta)))
        y_grass = np.concatenate(([0], 400 * np.cos(theta)))
        z_grass = np.zeros_like(x_grass)
        fig_3d.add_trace(go.Mesh3d(x=x_grass, y=y_grass, z=z_grass, color='#2e7d32', opacity=0.8, name='Outfield Grass', hoverinfo='skip'))
        
        # Infield Dirt (Mesh)
        theta_infield = np.linspace(-np.pi/4, np.pi/4, 30)
        x_dirt = np.concatenate(([0], 145 * np.sin(theta_infield)))
        y_dirt = np.concatenate(([0], 145 * np.cos(theta_infield)))
        z_dirt = np.full_like(x_dirt, 0.5) # slightly raised to prevent z-fighting
        fig_3d.add_trace(go.Mesh3d(x=x_dirt, y=y_dirt, z=z_dirt, color='#8d6e63', opacity=1.0, name='Infield Dirt', hoverinfo='skip'))

        # Foul Lines & Fence Boundary (Scatter3d)
        fig_3d.add_trace(go.Scatter3d(x=[0, 400*np.sin(np.pi/4)], y=[0, 400*np.cos(np.pi/4)], z=[1, 1], mode='lines', line=dict(color='white', width=6), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, 400*np.sin(-np.pi/4)], y=[0, 400*np.cos(-np.pi/4)], z=[1, 1], mode='lines', line=dict(color='white', width=6), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=400 * np.sin(theta), y=400 * np.cos(theta), z=np.ones_like(theta)*2, mode='lines', line=dict(color='#1b5e20', width=8), name='Fence'))
        
        # Bases
        bases_x = [63.6, 0, -63.6]
        bases_y = [63.6, 127.2, 63.6]
        fig_3d.add_trace(go.Scatter3d(x=bases_x, y=bases_y, z=[1,1,1], mode='markers', marker=dict(color='white', size=8, symbol='square'), name='Bases', hoverinfo='skip'))

        # Trajectories
        for idx, row in batted_balls.iterrows():
            dist, direction, angle, ev = row['Distance'], row['Direction'], row['Angle'], row['ExitSpeed']
            t = np.linspace(0, 1, 30)
            x_end = dist * np.sin(np.radians(direction))
            y_end = dist * np.cos(np.radians(direction))
            x_traj = t * x_end
            y_traj = t * y_end
            h_max = dist * np.tan(np.radians(max(0.1, angle))) * 0.25 
            z_traj = 4 * h_max * t * (1 - t)
            
            color = 'red' if angle > 25 else '#00e676' if angle >= 10 else '#29b6f6'
            fig_3d.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=z_traj, mode='lines', line=dict(color=color, width=4), hovertemplate=f"<b>EV:</b> {ev:.1f} mph<br><b>LA:</b> {angle:.1f}°<br><b>Dist:</b> {dist:.0f} ft<extra></extra>", showlegend=False))
            fig_3d.add_trace(go.Scatter3d(x=[x_end], y=[y_end], z=[0], mode='markers', marker=dict(color=color, size=4, symbol='circle'), showlegend=False))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='Horizontal (ft)', range=[-300, 300], showgrid=False, visible=False),
                yaxis=dict(title='Distance (ft)', range=[-20, 450], showgrid=False, visible=False),
                zaxis=dict(title='Height (ft)', range=[0, 150], showgrid=False, visible=False),
                aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.35),
                bgcolor='#e0f7fa' # light sky blue background
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=700
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        st.markdown("### 3D Contact Point & Volumetric Strike Zone")
        
        fig_cp = go.Figure()
        
        # 3D Home Plate
        plate_x = [-0.708, 0.708, 0.708, 0, -0.708, -0.708]
        plate_y = [0, 0, 0.708, 1.417, 0.708, 0] # Depth extending away from pitcher
        plate_z = [0, 0, 0, 0, 0, 0]
        fig_cp.add_trace(go.Scatter3d(x=plate_x, y=plate_y, z=plate_z, mode='lines', surfaceaxis=2, surfacecolor='white', line=dict(color='black', width=4), name='Home Plate', hoverinfo='skip'))
        
        # 3D Strike Zone Box (Wireframe)
        sz_x = [-0.833, 0.833, 0.833, -0.833, -0.833, -0.833, 0.833, 0.833, -0.833, -0.833]
        sz_y = [0, 0, 0, 0, 0, 1.417, 1.417, 1.417, 1.417, 1.417] # Front and back planes over the plate
        sz_z = [1.5, 1.5, 3.5, 3.5, 1.5, 1.5, 1.5, 3.5, 3.5, 1.5]
        
        # Connect front and back corners
        for x, z in [(-0.833, 1.5), (0.833, 1.5), (0.833, 3.5), (-0.833, 3.5)]:
            fig_cp.add_trace(go.Scatter3d(x=[x, x], y=[0, 1.417], z=[z, z], mode='lines', line=dict(color='rgba(255,0,0,0.4)', width=3), showlegend=False, hoverinfo='skip'))
        fig_cp.add_trace(go.Scatter3d(x=sz_x[:5], y=sz_y[:5], z=sz_z[:5], mode='lines', line=dict(color='rgba(255,0,0,0.6)', width=4), name='Front Zone (Pitcher Side)', hoverinfo='skip'))
        fig_cp.add_trace(go.Scatter3d(x=sz_x[5:], y=sz_y[5:], z=sz_z[5:], mode='lines', line=dict(color='rgba(255,0,0,0.4)', width=4), name='Back Zone (Catcher Side)', hoverinfo='skip'))

        # Plot Contact Points
        depth_col = 'ContactPositionZ' if 'ContactPositionZ' in batted_balls.columns else 'ContactPositionY'
        valid_cp = batted_balls.dropna(subset=['PlateLocSide', 'PlateLocHeight', depth_col])
        
        if not valid_cp.empty:
            fig_cp.add_trace(go.Scatter3d(
                x=valid_cp['PlateLocSide'], 
                y=valid_cp[depth_col], # Depth
                z=valid_cp['PlateLocHeight'],
                mode='markers',
                marker=dict(size=8, color=valid_cp['ExitSpeed'], colorscale='Viridis', showscale=True, colorbar=dict(title="EV (mph)"), line=dict(color='black', width=1)),
                hovertemplate="<b>EV:</b> %{marker.color:.1f} mph<br><b>X (Side):</b> %{x:.2f} ft<br><b>Y (Depth):</b> %{y:.2f} ft<br><b>Z (Height):</b> %{z:.2f} ft<extra></extra>",
                name='Contact Points'
            ))

        fig_cp.update_layout(
            scene=dict(
                xaxis=dict(title='Side to Side (ft)', range=[-3, 3]),
                yaxis=dict(title='Depth (ft)', range=[-1, 4]),
                zaxis=dict(title='Height (ft)', range=[0, 5]),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8),
            ),
            height=600, margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    with tab3:
        st.markdown("### Automated AI Player Profile & Scouting Report")
        if gemini_key:
            if st.button("Generate AI Scouting Report"):
                with st.spinner("Analyzing player data..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        batted_balls_copy = batted_balls.copy()
                        batted_balls_copy['SZ_Zone'] = batted_balls_copy.apply(lambda row: assign_sz_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
                        zone_evs = batted_balls_copy.groupby('SZ_Zone')['ExitSpeed'].mean().to_dict()
                        zone_las = batted_balls_copy.groupby('SZ_Zone')['Angle'].mean().to_dict()
                        
                        prompt = f"""
                        Act as a professional baseball hitting coordinator. Analyze the following Trackman batted ball data for {selected_batter} ({batter_side}). 
                        
                        Overall Metrics:
                        - Average Exit Velocity: {avg_ev:.1f} mph
                        - Max Exit Velocity: {max_ev:.1f} mph
                        - Hard Hit Percentage (>90% Max EV): {hh_pct:.1f}%
                        - Average Launch Angle: {avg_la:.1f} degrees
                        
                        Zone EV Averages (Z1-Z3 is Top, Z7-Z9 is Bottom):
                        {zone_evs}
                        
                        Zone Launch Angle Averages:
                        {zone_las}
                        
                        Provide a concise, 3-paragraph scouting report identifying strengths, specific pitches/zones they crush, and mechanical flags based on zone performance.
                        """
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")
        else:
            st.warning("Please enter your Gemini API Key in the sidebar to unlock AI Scouting Analysis.")

    with tab4:
        st.markdown("### Export Professional Analytics Report")
        
        def create_full_pdf_report(df, batter_name, batter_side, session_date):
            pdf_buffer = BytesIO()
            fig = plt.figure(figsize=(14, 11)) 
            
            fig.patches.append(Rectangle((0.015, 0.015), 0.97, 0.97, fill=False, edgecolor='black', lw=1, transform=fig.transFigure))
            fig.text(0.04, 0.94, f"{batter_name}  |  {batter_side}", fontsize=26, fontweight='bold', color='#0f172a')
            fig.text(0.04, 0.91, f"OMAHA PLAYER DEVELOPMENT • Profile Date: {session_date}", fontsize=14, color='#64748b')
            fig.add_artist(plt.Line2D((0.04, 0.89), (0.89, 0.89), color='#cbd5e1', linewidth=1.5))

            box_width = 0.075 
            x_offsets = np.linspace(0.025, 0.885, 10).tolist()
            pdf_metrics = [
                ("SWINGS", f"{int(total_swings)}"), ("TOTAL HH", f"{int(total_hh)}"), ("HARD HIT %", f"{hh_pct:.1f}%"),
                ("AVG EV", f"{avg_ev:.1f} mph"), ("MAX EV", f"{max_ev:.1f} mph"), ("AVG LA", f"{avg_la:.1f}°"),
                ("AVG DIST", f"{avg_dist:.0f} ft"), ("MAX DIST", f"{max_dist:.0f} ft"), ("AVG SPIN", f"{avg_spin:.0f} rpm"), ("AVG HANG", f"{avg_hang:.2f} s")
            ]
            
            for idx, (label, val) in enumerate(pdf_metrics):
                fig.patches.append(FancyBboxPatch((x_offsets[idx], 0.80), box_width, 0.06, boxstyle="round,pad=0.005", fc="#f8fafc", ec="#cbd5e1", lw=1.5, transform=fig.transFigure))
                cx = x_offsets[idx] + (box_width / 2)
                fig.text(cx, 0.84, label, fontsize=7.5, fontweight='bold', color='#64748b', ha='center')
                fig.text(cx, 0.815, val, fontsize=12, fontweight='bold', color='#0f172a', ha='center')

            # Row 1 Charts
            ax1_pdf = fig.add_axes([0.025, 0.42, 0.28, 0.34])
            draw_field_skeleton(ax1_pdf, draw_infield=True)
            gb, ld, fb = df[df['Angle'] < 10], df[(df['Angle'] >= 10) & (df['Angle'] <= 25)], df[df['Angle'] > 25]
            for sub_df, color, lbl in [(gb, '#3b82f6', 'GB (<10°)'), (ld, '#22c55e', 'LD (10°-25°)'), (fb, '#ef4444', 'FB (>25°)')]:
                if not sub_df.empty: ax1_pdf.scatter(sub_df['Distance'] * np.sin(np.radians(sub_df['Direction'])), sub_df['Distance'] * np.cos(np.radians(sub_df['Direction'])), color=color, s=45, edgecolors='#0f172a', linewidths=0.5, label=lbl, zorder=6)
            ax1_pdf.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Batted Ball Profile")
            ax1_pdf.legend(loc='lower left', frameon=True, fontsize=8)

            ax2_pdf = fig.add_axes([0.345, 0.42, 0.28, 0.34])
            draw_zoned_plot(ax2_pdf, df, metric='ExitSpeed', title="Avg EV by Field Quadrant", vmin=75, vmax=100, cmap='coolwarm', unit='', is_ev=True)

            ax3_pdf = fig.add_axes([0.665, 0.42, 0.28, 0.34])
            draw_zoned_plot(ax3_pdf, df, metric='Angle', title="Avg LA by Field Quadrant", vmin=0, vmax=35, cmap='coolwarm', unit='°')

            # Row 2 Charts
            ax4_pdf = fig.add_axes([0.025, 0.06, 0.28, 0.34])
            draw_strike_zone_plot(ax4_pdf, df, metric='ExitSpeed', title="Avg EV by Pitch Location", vmin=75, vmax=100, cmap='coolwarm', is_ev=True)
            
            ax5_pdf = fig.add_axes([0.345, 0.06, 0.28, 0.34])
            draw_strike_zone_plot(ax5_pdf, df, metric='Angle', title="Avg LA by Pitch Location", vmin=0, vmax=35, cmap='coolwarm', unit='°')

            ax6_pdf = fig.add_axes([0.665, 0.06, 0.28, 0.34])
            draw_contact_plot(ax6_pdf, df, title="Contact Depth Position")

            fig.savefig(pdf_buffer, format='pdf', dpi=150)
            plt.close(fig)
            return pdf_buffer.getvalue()

        st.info("The exported PDF will include the complete 6-chart layout exactly as specified in your original requirements, incorporating all underlying session parameters.")
        pdf_bytes = create_full_pdf_report(batted_balls, selected_batter, batter_side, selected_date)
        safe_name = selected_batter.replace(", ", "_").replace(" ", "_")
        st.download_button(
            label=f"📄 Download Complete {selected_batter} Report",
            data=pdf_bytes,
            file_name=f"Omaha_BP_Analytics_{safe_name}.pdf",
            mime="application/pdf"
        )
