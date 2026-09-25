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

# --- PAGE SETUP & PREMIUM THEMING ---
st.set_page_config(page_title="Omaha Player Development Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .kpi-card {
            background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 18px 10px; text-align: center; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); 
            margin-bottom: 20px; transition: transform 0.2s;
        }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
        .kpi-label { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-value { font-size: 24px; font-weight: 900; color: #0f172a; margin-top: 4px; }
        .header-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px; padding: 24px 30px; color: white; margin-bottom: 25px;
            border-left: 6px solid #e02424; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 32px; font-weight: 900; margin: 0; letter-spacing: -0.5px; }
        .header-sub { font-size: 15px; font-weight: 500; color: #94a3b8; margin-top: 6px; letter-spacing: 1px; text-transform: uppercase; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 10px 20px; font-weight: 600; }
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
        if z >= 2.833: return 'Z1' if x <= -0.277 else 'Z2' if x <= 0.277 else 'Z3'
        elif z >= 2.166: return 'Z4' if x <= -0.277 else 'Z5' if x <= 0.277 else 'Z6'
        else: return 'Z7' if x <= -0.277 else 'Z8' if x <= 0.277 else 'Z9'
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
        color = colormap(norm(avg_val)) if not pd.isna(avg_val) else '#f8fafc'
        text_val = f"{avg_val:.1f}{unit}" if not pd.isna(avg_val) else "N/A"
        font_size = 8 if is_ev and zone_name.startswith('IF') else 11
        ax.add_patch(Wedge(center=(0, 0), r=config['r_inner'] + config['width'], theta1=config['theta1'], theta2=config['theta2'], width=config['width'], facecolor=color, edgecolor='#cbd5e1', lw=1.5, alpha=0.85, zorder=1))
        text_x, text_y = config['text_r'] * np.cos(np.radians(config['text_theta'])), config['text_r'] * np.sin(np.radians(config['text_theta']))
        ax.text(text_x, text_y, text_val, ha='center', va='center', fontsize=font_size, fontweight='bold', color='#0f172a', bbox=dict(facecolor='white', alpha=0.85, edgecolor='#e2e8f0', boxstyle='round,pad=0.3'), zorder=4)
    ax.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title=title)

def draw_strike_zone_plot(ax, df, metric, title, vmin, vmax, cmap='coolwarm', unit='', is_ev=False):
    df_copy = df.copy()
    df_copy['SZ_Zone'] = df_copy.apply(lambda row: assign_sz_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
    zone_avgs = df_copy.groupby('SZ_Zone')[metric].mean()
    norm, colormap = mcolors.Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap)
    for zone, rect in SZ_ZONES_CONFIG.items():
        avg_val = zone_avgs.get(zone, np.nan)
        color = colormap(norm(avg_val)) if not pd.isna(avg_val) else '#f8fafc'
        text_val = (f"{avg_val:.1f}{unit}" if not is_ev else f"{avg_val:.1f}") if not pd.isna(avg_val) else "N/A"
        ax.add_patch(Rectangle((rect['x'], rect['y']), rect['w'], rect['h'], facecolor=color, edgecolor='#94a3b8', lw=1.5, zorder=1))
        font_size = 10 if zone.startswith('Z') else 8
        ax.text(rect['x'] + rect['w']/2, rect['y'] + rect['h']/2, text_val, ha='center', va='center', fontsize=font_size, fontweight='bold', color='#0f172a', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=3)
    ax.add_patch(Rectangle((-0.833, 1.5), 1.666, 2.0, fill=False, edgecolor='black', lw=2.5, zorder=2))
    ax.add_patch(MplPolygon(np.column_stack(([-0.708, 0.708, 0.708, 0, -0.708], [0, 0, 0.25, 0.5, 0.25])), facecolor='white', edgecolor='black', lw=1.5, zorder=2))
    ax.set(xlim=(-2, 2), ylim=(0, 4.5), aspect='equal', xticks=[], yticks=[], title=title)

def draw_contact_plot(ax, df, title="Contact Depth Position"):
    df_cp = df.copy()
    df_cp['CP_X_in'] = df_cp['ContactPositionX'] * 12 if 'ContactPositionX' in df_cp.columns else df_cp['PlateLocSide'] * 12
    depth_col = 'ContactPositionZ' if 'ContactPositionZ' in df_cp.columns else 'ContactPositionY'
    df_cp['CP_Z_in'] = df_cp[depth_col] * 12 if depth_col in df_cp.columns else 0
    
    ax.grid(True, linestyle='--', alpha=0.6, color='#cbd5e1', zorder=0)
    ax.axhline(0, color='black', lw=1.5, zorder=1)
    ax.axvline(0, color='black', lw=1.5, zorder=1)
    ax.add_patch(MplPolygon(np.column_stack(([0, -8.5, -8.5, 8.5, 8.5, 0], [0, 8.5, 17, 17, 8.5, 0])), facecolor='#f1f5f9', edgecolor='black', lw=1.5, zorder=2))
    ax.scatter(df_cp['CP_Z_in'], df_cp['CP_X_in'], c=df_cp['ExitSpeed'], cmap='coolwarm', vmin=75, vmax=100, s=65, edgecolors='#0f172a', lw=0.75, zorder=5)
    ax.set(xlim=(-30, 30), ylim=(-5, 55), aspect='equal', title=title, xlabel="Side to Side (inches)", ylabel="Out in Front (inches)")

# --- SIDEBAR & FILTERING ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Omaha_Mavericks_logo.svg/1200px-Omaha_Mavericks_logo.svg.png", width=140)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📥 Data Management")
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
            
            if st.button("Save to Database", type="primary", use_container_width=True):
                success, msg = load_data_to_db(df_cleaned)
                if success: st.success(msg)
                else: st.warning(msg)
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Dashboard Filters")
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
        st.info("Database is empty. Please upload a Trackman CSV to begin.")
        batted_balls = pd.DataFrame()
        selected_batter = None

    st.markdown("---")
    st.markdown("### 🧠 API Integration")
    gemini_key = st.text_input("Gemini API Key (For AI Scouting)", type="password")

# --- MAIN DASHBOARD ---
if not batted_balls.empty and selected_batter:
    # Key Metrics Calculation
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
    
    # Premium Header
    st.markdown(f"""
        <div class="header-box">
            <div class="header-title">{selected_batter} <span style="color:#ef4444; font-weight:300;">|</span> <span style="color:#cbd5e1; font-weight:600; font-size:26px;">{batter_side}</span></div>
            <div class="header-sub">Omaha Player Development Analytics • Date Range: {selected_date}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Grid
    kpi_cols = st.columns(10)
    kpis = [
        ("Swings", int(total_swings)), ("Balls In Play", int(total_bip)), ("Hard Hit %", f"{hh_pct:.1f}%"), 
        ("Avg EV", f"{avg_ev:.1f}"), ("Max EV", f"{max_ev:.1f}"), ("Avg LA", f"{avg_la:.1f}°"), 
        ("Avg Dist", f"{avg_dist:.0f}"), ("Max Dist", f"{max_dist:.0f}"), ("Avg Spin", f"{avg_spin:.0f}"), ("Hang", f"{avg_hang:.2f}s")
    ]
    for col, (label, val) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    # Tabs Configuration
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Elite Field Trajectories", "⚾ 3D Strike Zone & Contact", "🧠 Automated AI Scouting", "📄 PDF Reports & Raw Data"])
    
    # --- TAB 1: ELITE 3D FIELD ---
    with tab1:
        st.markdown("<h3 style='color:#0f172a; margin-bottom: 20px;'>Interactive 3D Spray Chart</h3>", unsafe_allow_html=True)
        fig_3d = go.Figure()
        
        # 1. Mowed Grass Outfield (Alternating Stripes)
        theta_arc = np.linspace(-np.pi/4, np.pi/4, 150)
        for r in range(150, 401, 25):
            x_out, y_out = r * np.sin(theta_arc), r * np.cos(theta_arc)
            x_in, y_in = (r-25) * np.sin(theta_arc[::-1]), (r-25) * np.cos(theta_arc[::-1])
            stripe_color = '#2e7d32' if (r//25)%2 == 0 else '#388e3c'
            fig_3d.add_trace(go.Scatter3d(
                x=np.concatenate([x_out, x_in, [x_out[0]]]), 
                y=np.concatenate([y_out, y_in, [y_out[0]]]), 
                z=np.zeros(301),
                mode='lines', surfaceaxis=2, surfacecolor=stripe_color, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ))
            
        # 2. Infield Dirt Cutout
        theta_infield = np.linspace(-np.pi/4, np.pi/4, 50)
        fig_3d.add_trace(go.Scatter3d(
            x=np.concatenate([[0], 145 * np.sin(theta_infield), [0]]),
            y=np.concatenate([[0], 145 * np.cos(theta_infield), [0]]),
            z=np.full(52, 0.1), mode='lines', surfaceaxis=2, surfacecolor='#a1887f', line=dict(width=0), hoverinfo='skip', showlegend=False
        ))
        
        # 3. Warning Track
        fig_3d.add_trace(go.Scatter3d(
            x=np.concatenate([400 * np.sin(theta_arc), 385 * np.sin(theta_arc[::-1]), [400 * np.sin(theta_arc[0])]]),
            y=np.concatenate([400 * np.cos(theta_arc), 385 * np.cos(theta_arc[::-1]), [400 * np.cos(theta_arc[0])]]),
            z=np.full(301, 0.2), mode='lines', surfaceaxis=2, surfacecolor='#8d6e63', line=dict(width=0), hoverinfo='skip', showlegend=False
        ))

        # 4. Bases & Foul Lines
        bases_x, bases_y = [63.6, 0, -63.6], [63.6, 127.2, 63.6]
        fig_3d.add_trace(go.Scatter3d(x=bases_x, y=bases_y, z=[0.3, 0.3, 0.3], mode='markers', marker=dict(color='white', size=6, symbol='square'), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, 283], y=[0, 283], z=[0.1, 0.1], mode='lines', line=dict(color='white', width=3), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, -283], y=[0, 283], z=[0.1, 0.1], mode='lines', line=dict(color='white', width=3), hoverinfo='skip', showlegend=False))

        # 5. Outfield Wall & Foul Poles
        fig_3d.add_trace(go.Scatter3d(x=400 * np.sin(theta_arc), y=400 * np.cos(theta_arc), z=np.full(150, 10), mode='lines', surfaceaxis=2, surfacecolor='#0f172a', line=dict(width=4, color='#1e293b'), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[283, 283], y=[283, 283], z=[0, 40], mode='lines', line=dict(color='yellow', width=6), hoverinfo='skip', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[-283, -283], y=[283, 283], z=[0, 40], mode='lines', line=dict(color='yellow', width=6), hoverinfo='skip', showlegend=False))

        # 6. Smooth Trajectories & Ground Shadows
        for idx, row in batted_balls.iterrows():
            dist, direction, angle, ev = row['Distance'], row['Direction'], row['Angle'], row['ExitSpeed']
            t = np.linspace(0, 1, 50)
            x_end, y_end = dist * np.sin(np.radians(direction)), dist * np.cos(np.radians(direction))
            x_traj, y_traj = t * x_end, t * y_end
            h_max = dist * np.tan(np.radians(max(0.1, angle))) * 0.25 
            z_traj = 4 * h_max * t * (1 - t)
            
            # Color logic based on Launch Angle
            if angle > 25: traj_color = '#ef4444' # Flyball (Red)
            elif angle >= 10: traj_color = '#3b82f6' # Line Drive (Blue)
            else: traj_color = '#eab308' # Groundball (Yellow)
            
            # Airborne Path
            fig_3d.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=z_traj, mode='lines', line=dict(color=traj_color, width=4), hovertemplate=f"<b>EV:</b> {ev:.1f} mph<br><b>LA:</b> {angle:.1f}°<br><b>Dist:</b> {dist:.0f} ft<extra></extra>", showlegend=False))
            # Ground Shadow
            fig_3d.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=np.zeros(50), mode='lines', line=dict(color='rgba(0,0,0,0.4)', width=2, dash='dash'), hoverinfo='skip', showlegend=False))
            # Landing Marker
            fig_3d.add_trace(go.Scatter3d(x=[x_end], y=[y_end], z=[0], mode='markers', marker=dict(color=traj_color, size=5, line=dict(color='white', width=1)), hoverinfo='skip', showlegend=False))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(range=[-300, 300], visible=False),
                yaxis=dict(range=[-20, 450], visible=False),
                zaxis=dict(range=[0, 150], visible=False),
                aspectmode='manual', aspectratio=dict(x=1.3, y=1.3, z=0.35),
                camera=dict(eye=dict(x=0, y=-1.5, z=0.8)) # Default isometric view behind home plate
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=750, paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 2: ELITE 3D STRIKE ZONE ---
    with tab2:
        st.markdown("<h3 style='color:#0f172a; margin-bottom: 20px;'>Volumetric Contact Analysis</h3>", unsafe_allow_html=True)
        fig_cp = go.Figure()
        
        # 3D Home Plate
        plate_x, plate_y, plate_z = [-0.708, 0.708, 0.708, 0, -0.708, -0.708], [0, 0, 0.708, 1.417, 0.708, 0], [0, 0, 0, 0, 0, 0]
        fig_cp.add_trace(go.Scatter3d(x=plate_x, y=plate_y, z=plate_z, mode='lines', surfaceaxis=2, surfacecolor='white', line=dict(color='#0f172a', width=5), hoverinfo='skip', showlegend=False))
        
        # 3D Strike Zone Box (Glass Frame)
        sz_x = [-0.833, 0.833, 0.833, -0.833, -0.833, -0.833, 0.833, 0.833, -0.833, -0.833]
        sz_y = [0, 0, 0, 0, 0, 1.417, 1.417, 1.417, 1.417, 1.417]
        sz_z = [1.5, 1.5, 3.5, 3.5, 1.5, 1.5, 1.5, 3.5, 3.5, 1.5]
        
        for x, z in [(-0.833, 1.5), (0.833, 1.5), (0.833, 3.5), (-0.833, 3.5)]:
            fig_cp.add_trace(go.Scatter3d(x=[x, x], y=[0, 1.417], z=[z, z], mode='lines', line=dict(color='rgba(226, 232, 240, 0.8)', width=3), hoverinfo='skip', showlegend=False))
        fig_cp.add_trace(go.Scatter3d(x=sz_x[:5], y=sz_y[:5], z=sz_z[:5], mode='lines', line=dict(color='rgba(59, 130, 246, 0.8)', width=5), name='Pitcher Side Zone', hoverinfo='skip'))
        fig_cp.add_trace(go.Scatter3d(x=sz_x[5:], y=sz_y[5:], z=sz_z[5:], mode='lines', line=dict(color='rgba(148, 163, 184, 0.5)', width=4), name='Catcher Side Zone', hoverinfo='skip'))

        # Plotted Contact Points
        depth_col = 'ContactPositionZ' if 'ContactPositionZ' in batted_balls.columns else 'ContactPositionY'
        valid_cp = batted_balls.dropna(subset=['PlateLocSide', 'PlateLocHeight', depth_col])
        
        if not valid_cp.empty:
            fig_cp.add_trace(go.Scatter3d(
                x=valid_cp['PlateLocSide'], y=valid_cp[depth_col], z=valid_cp['PlateLocHeight'],
                mode='markers',
                marker=dict(size=8, color=valid_cp['ExitSpeed'], colorscale='Turbo', showscale=True, colorbar=dict(title="Exit Velo", len=0.7), line=dict(color='black', width=1)),
                hovertemplate="<b>EV:</b> %{marker.color:.1f} mph<br><b>X (Side):</b> %{x:.2f} ft<br><b>Y (Depth):</b> %{y:.2f} ft<br><b>Z (Height):</b> %{z:.2f} ft<extra></extra>",
                name='Contact Depth'
            ))

        fig_cp.update_layout(
            scene=dict(
                xaxis=dict(title='Side to Side (ft)', range=[-3, 3], backgroundcolor="#f8fafc", gridcolor="white", showbackground=True),
                yaxis=dict(title='Depth from Pitcher (ft)', range=[-1, 4], backgroundcolor="#f8fafc", gridcolor="white", showbackground=True),
                zaxis=dict(title='Height (ft)', range=[0, 5], backgroundcolor="#f8fafc", gridcolor="white", showbackground=True),
                aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.9),
            ),
            height=650, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    # --- TAB 3: AI SCOUTING ---
    with tab3:
        st.markdown("<h3 style='color:#0f172a; margin-bottom: 20px;'>Automated Profile Synthesis</h3>", unsafe_allow_html=True)
        if gemini_key:
            if st.button("Generate Scouting Report", type="primary"):
                with st.spinner("Analyzing player data metrics..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        batted_balls_copy = batted_balls.copy()
                        batted_balls_copy['SZ_Zone'] = batted_balls_copy.apply(lambda row: assign_sz_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
                        zone_evs = batted_balls_copy.groupby('SZ_Zone')['ExitSpeed'].mean().to_dict()
                        zone_las = batted_balls_copy.groupby('SZ_Zone')['Angle'].mean().to_dict()
                        
                        prompt = f"""
                        Act as an elite D1/Pro baseball hitting coordinator. Analyze Trackman batted ball data for {selected_batter} ({batter_side}). 
                        
                        Session Metrics:
                        - Avg EV: {avg_ev:.1f} mph
                        - Max EV: {max_ev:.1f} mph
                        - Hard Hit % (>90% Max EV): {hh_pct:.1f}%
                        - Avg Launch Angle: {avg_la:.1f}°
                        
                        Zone EV Averages (Z1-Z3=Top, Z7-Z9=Bottom): {zone_evs}
                        Zone Launch Angle Averages: {zone_las}
                        
                        Provide a concise, 3-paragraph scouting report identifying 1) overall power/bat-speed profile, 2) specific pitch zones they excel in, and 3) developmental flags/deficiencies based on zone performance.
                        """
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")
        else:
            st.warning("Please enter your Gemini API Key in the sidebar to run AI analysis.")

    # --- TAB 4: PDF REPORT & RAW DATA ---
    with tab4:
        st.markdown("<h3 style='color:#0f172a; margin-bottom: 20px;'>Export Executive Report</h3>", unsafe_allow_html=True)
        
        def create_full_pdf_report(df, batter_name, batter_side, session_date):
            pdf_buffer = BytesIO()
            fig = plt.figure(figsize=(14, 11)) 
            fig.patches.append(Rectangle((0.015, 0.015), 0.97, 0.97, fill=False, edgecolor='#0f172a', lw=2, transform=fig.transFigure))
            
            # PDF Header
            fig.text(0.04, 0.94, f"{batter_name}  |  {batter_side}", fontsize=28, fontweight='900', color='#0f172a')
            fig.text(0.04, 0.91, f"OMAHA PLAYER DEVELOPMENT • Profile Date: {session_date}", fontsize=14, color='#64748b', fontweight='bold')
            fig.add_artist(plt.Line2D((0.04, 0.89), (0.89, 0.89), color='#e2e8f0', linewidth=2))

            # PDF Metrics
            box_width = 0.075 
            x_offsets = np.linspace(0.025, 0.885, 10).tolist()
            pdf_metrics = [
                ("SWINGS", f"{int(total_swings)}"), ("TOTAL HH", f"{int(total_hh)}"), ("HARD HIT %", f"{hh_pct:.1f}%"),
                ("AVG EV", f"{avg_ev:.1f} mph"), ("MAX EV", f"{max_ev:.1f} mph"), ("AVG LA", f"{avg_la:.1f}°"),
                ("AVG DIST", f"{avg_dist:.0f} ft"), ("MAX DIST", f"{max_dist:.0f} ft"), ("AVG SPIN", f"{avg_spin:.0f} rpm"), ("AVG HANG", f"{avg_hang:.2f} s")
            ]
            for idx, (label, val) in enumerate(pdf_metrics):
                fig.patches.append(FancyBboxPatch((x_offsets[idx], 0.80), box_width, 0.06, boxstyle="round,pad=0.01", fc="#f8fafc", ec="#cbd5e1", lw=1.5, transform=fig.transFigure))
                cx = x_offsets[idx] + (box_width / 2)
                fig.text(cx, 0.84, label, fontsize=8, fontweight='bold', color='#64748b', ha='center')
                fig.text(cx, 0.815, val, fontsize=12, fontweight='bold', color='#0f172a', ha='center')

            # Row 1 Matplotlib Charts (2D equivalent for PDF)
            ax1_pdf = fig.add_axes([0.025, 0.42, 0.28, 0.34])
            draw_field_skeleton(ax1_pdf, draw_infield=True)
            gb, ld, fb = df[df['Angle'] < 10], df[(df['Angle'] >= 10) & (df['Angle'] <= 25)], df[df['Angle'] > 25]
            for sub_df, color, lbl in [(gb, '#eab308', 'GB (<10°)'), (ld, '#3b82f6', 'LD (10°-25°)'), (fb, '#ef4444', 'FB (>25°)')]:
                if not sub_df.empty: ax1_pdf.scatter(sub_df['Distance'] * np.sin(np.radians(sub_df['Direction'])), sub_df['Distance'] * np.cos(np.radians(sub_df['Direction'])), color=color, s=50, edgecolors='#0f172a', linewidths=0.75, label=lbl, zorder=6)
            ax1_pdf.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Batted Ball Profile")
            ax1_pdf.legend(loc='lower left', frameon=True, fontsize=9)

            ax2_pdf = fig.add_axes([0.345, 0.42, 0.28, 0.34])
            draw_zoned_plot(ax2_pdf, df, metric='ExitSpeed', title="Avg EV by Field Quadrant", vmin=75, vmax=100, cmap='coolwarm', unit='', is_ev=True)

            ax3_pdf = fig.add_axes([0.665, 0.42, 0.28, 0.34])
            draw_zoned_plot(ax3_pdf, df, metric='Angle', title="Avg LA by Field Quadrant", vmin=0, vmax=35, cmap='coolwarm', unit='°')

            # Row 2 Matplotlib Charts
            ax4_pdf = fig.add_axes([0.025, 0.06, 0.28, 0.34])
            draw_strike_zone_plot(ax4_pdf, df, metric='ExitSpeed', title="Avg EV by Pitch Location", vmin=75, vmax=100, cmap='coolwarm', is_ev=True)
            
            ax5_pdf = fig.add_axes([0.345, 0.06, 0.28, 0.34])
            draw_strike_zone_plot(ax5_pdf, df, metric='Angle', title="Avg LA by Pitch Location", vmin=0, vmax=35, cmap='coolwarm', unit='°')

            ax6_pdf = fig.add_axes([0.665, 0.06, 0.28, 0.34])
            draw_contact_plot(ax6_pdf, df, title="Contact Depth Position")

            fig.savefig(pdf_buffer, format='pdf', dpi=150)
            plt.close(fig)
            return pdf_buffer.getvalue()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.info("Export a formatted, 6-chart PDF document containing the player's underlying parameters for this specific selection.")
            pdf_bytes = create_full_pdf_report(batted_balls, selected_batter, batter_side, selected_date)
            safe_name = selected_batter.replace(", ", "_").replace(" ", "_")
            st.download_button(
                label=f"📄 Download Profile PDF",
                data=pdf_bytes,
                file_name=f"Omaha_Development_Report_{safe_name}.pdf",
                mime="application/pdf",
                type="primary"
            )
        
        st.markdown("<br><h5>Raw Batted Ball Data</h5>", unsafe_allow_html=True)
        st.dataframe(batted_balls[['PitchNo', 'Pitcher', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'HitSpinRate', 'PlateLocHeight', 'PlateLocSide']].sort_values(by='ExitSpeed', ascending=False), use_container_width=True)
