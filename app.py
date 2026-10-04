import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Wedge, FancyBboxPatch
import matplotlib.colors as mcolors
from io import BytesIO
import base64
import io

# ==========================================
# 1. SYSTEM CONFIGURATION & INITIALIZATION
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE, "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800;900&display=swap"], suppress_callback_exceptions=True)
app.title = "Omaha Baseball Analytics"
server = app.server

DB_NAME = "trackman_master.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trackman_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SessionType TEXT, Season TEXT, Opponent TEXT,
            PitchNo INTEGER, Date TEXT, Time TEXT, Pitcher TEXT, Batter TEXT, BatterSide TEXT, PitcherThrows TEXT,
            TaggedPitchType TEXT, ExitSpeed REAL, Angle REAL, Direction REAL, Distance REAL, HitSpinRate REAL, HangTime REAL,
            PlateLocHeight REAL, PlateLocSide REAL, ContactPositionX REAL, ContactPositionY REAL, ContactPositionZ REAL,
            PitchCall TEXT, PlayResult TEXT, TaggedHitType TEXT, KorBB TEXT,
            Balls INTEGER, Strikes INTEGER, RelSpeed REAL, InducedVertBreak REAL, HorzBreak REAL, VertApprAngle REAL,
            UploadTimestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. ADVANCED ANALYTICS & MATH ENGINE
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
    return 'Out'

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

def calc_batting_line(df):
    if df.empty:
        return pd.DataFrame([{"Split": "TOTAL", "AB": 0, "H": 0, "1B": 0, "2B": 0, "3B": 0, "HR": 0, "BB": 0, "K": 0, "AVG": "-", "SLG": "-", "OBP": "-", "OPS": "-"}])
    
    df['PlayResult'] = df['PlayResult'].fillna('')
    df['KorBB'] = df['KorBB'].fillna('')
    df['PitchCall'] = df['PitchCall'].fillna('')
    
    h1 = len(df[df['PlayResult'] == 'Single'])
    h2 = len(df[df['PlayResult'] == 'Double'])
    h3 = len(df[df['PlayResult'] == 'Triple'])
    hr = len(df[df['PlayResult'] == 'HomeRun'])
    hits = h1 + h2 + h3 + hr
    
    k = len(df[df['KorBB'] == 'Strikeout'])
    bb = len(df[df['KorBB'] == 'Walk'])
    hbp = len(df[df['PitchCall'] == 'HitByPitch'])
    sac = len(df[df['PlayResult'] == 'Sacrifice'])
    
    outs = len(df[df['PlayResult'].isin(['Out', 'Error', 'FieldersChoice'])])
    abs_count = hits + outs + k
    
    tb = h1 + (h2 * 2) + (h3 * 3) + (hr * 4)
    obp_den = abs_count + bb + hbp + sac
    
    avg = hits / abs_count if abs_count > 0 else 0
    slg = tb / abs_count if abs_count > 0 else 0
    obp = (hits + bb + hbp) / obp_den if obp_den > 0 else 0
    ops = obp + slg
    
    return pd.DataFrame([{
        "AB": abs_count, "H": hits, "1B": h1, "2B": h2, "3B": h3, "HR": hr, "BB": bb, "K": k,
        "AVG": f"{avg:.3f}".lstrip('0'), "SLG": f"{slg:.3f}".lstrip('0'), "OBP": f"{obp:.3f}".lstrip('0'), "OPS": f"{ops:.3f}".lstrip('0')
    }])

def get_full_batting_line(df):
    total = calc_batting_line(df)
    total['Split'] = "TOTAL"
    lhp = calc_batting_line(df[df['PitcherThrows'] == 'Left'])
    lhp['Split'] = "Vs. LHP"
    rhp = calc_batting_line(df[df['PitcherThrows'] == 'Right'])
    rhp['Split'] = "Vs. RHP"
    combined = pd.concat([total, lhp, rhp], ignore_index=True)
    return combined[['Split', 'AB', 'H', '1B', '2B', '3B', 'HR', 'BB', 'K', 'AVG', 'SLG', 'OBP', 'OPS']]

# ==========================================
# 3. PLOTLY RENDERING ENGINES
# ==========================================
def render_3d_stadium(df):
    fig = go.Figure()
    grass_material = dict(ambient=0.8, diffuse=0.9, roughness=0.9, specular=0.1)
    dirt_material = dict(ambient=0.9, diffuse=0.8, roughness=1.0, specular=0.0)

    # Majestic Outfield Grass Striping
    theta_arc = np.linspace(-np.pi/4, np.pi/4, 200)
    for r in range(150, 401, 25):
        x_out, y_out = r * np.sin(theta_arc), r * np.cos(theta_arc)
        x_in, y_in = (r-25) * np.sin(theta_arc[::-1]), (r-25) * np.cos(theta_arc[::-1])
        stripe_color = '#14532D' if (r//25)%2 == 0 else '#166534'
        fig.add_trace(go.Mesh3d(x=np.concatenate([x_out, x_in, [x_out[0]]]), y=np.concatenate([y_out, y_in, [y_out[0]]]), z=np.zeros(401), color=stripe_color, lighting=grass_material, hoverinfo='skip', showlegend=False))
        
    # High-Definition Infield Dirt
    theta_full = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Mesh3d(x=np.concatenate([[0], 145 * np.sin(np.linspace(-np.pi/4, np.pi/4, 100)), [0]]), y=np.concatenate([[0], 145 * np.cos(np.linspace(-np.pi/4, np.pi/4, 100)), [0]]), z=np.full(102, 0.1), color='#8D6E63', lighting=dirt_material, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Mesh3d(x=13*np.sin(theta_full), y=13*np.cos(theta_full), z=np.full(100, 0.15), color='#8D6E63', lighting=dirt_material, hoverinfo='skip', showlegend=False)) 
    fig.add_trace(go.Mesh3d(x=9*np.sin(theta_full), y=60.5 + 9*np.cos(theta_full), z=np.full(100, 0.15), color='#8D6E63', lighting=dirt_material, hoverinfo='skip', showlegend=False)) 

    # Warning Track
    fig.add_trace(go.Mesh3d(x=np.concatenate([400 * np.sin(theta_arc), 385 * np.sin(theta_arc[::-1]), [400 * np.sin(theta_arc[0])]]), y=np.concatenate([400 * np.cos(theta_arc), 385 * np.cos(theta_arc[::-1]), [400 * np.cos(theta_arc[0])]]), z=np.full(401, 0.2), color='#5D4037', lighting=dirt_material, hoverinfo='skip', showlegend=False))
    
    # Ultra-Professional Translucent Outfield Wall
    wall_x = np.concatenate([400 * np.sin(theta_arc), 400 * np.sin(theta_arc[::-1])])
    wall_y = np.concatenate([400 * np.cos(theta_arc), 400 * np.cos(theta_arc[::-1])])
    wall_z = np.concatenate([np.zeros(200), np.full(200, 15)])
    fig.add_trace(go.Mesh3d(x=wall_x, y=wall_y, z=wall_z, color='rgba(15, 23, 42, 0.4)', alphahull=0, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter3d(x=400 * np.sin(theta_arc), y=400 * np.cos(theta_arc), z=np.full(200, 15), mode='lines', line=dict(color='#D71920', width=8), hoverinfo='skip', showlegend=False))

    # Base Paths & Chalk Lines
    fig.add_trace(go.Scatter3d(x=[63.6, 0, -63.6], y=[63.6, 127.2, 63.6], z=[0.3, 0.3, 0.3], mode='markers', marker=dict(color='white', size=6, symbol='square'), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 283], y=[0, 283], z=[0.2, 0.2], mode='lines', line=dict(color='white', width=4), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, -283], y=[0, 283], z=[0.2, 0.2], mode='lines', line=dict(color='white', width=4), hoverinfo='skip', showlegend=False))

    if not df.empty:
        for _, row in df.iterrows():
            dist, direction, angle, ev = row['Distance'], row['Direction'], row['Angle'], row['ExitSpeed']
            if pd.isna(dist) or pd.isna(direction) or pd.isna(angle): continue
            t = np.linspace(0, 1, 60)
            x_end, y_end = dist * np.sin(np.radians(direction)), dist * np.cos(np.radians(direction))
            x_traj, y_traj = t * x_end, t * y_end
            h_max = dist * np.tan(np.radians(max(0.1, angle))) * 0.25 
            z_traj = 4 * h_max * t * (1 - t)
            
            color = '#D71920' if (ev >= 95 and 8 <= angle <= 32) else '#3B82F6' if ev >= 90 else '#94A3B8'
            
            # Interactive Flight Path
            fig.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=z_traj, mode='lines', line=dict(color=color, width=6), hovertemplate=f"<b>EV:</b> {ev:.1f} mph<br><b>LA:</b> {angle:.1f}°<br><b>Dist:</b> {dist:.0f} ft<extra></extra>", showlegend=False))
            # Ground Tracking Shadow
            fig.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=np.full(60, 0.3), mode='lines', line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dot'), hoverinfo='skip', showlegend=False))
            # Baseball Landing Marker
            fig.add_trace(go.Scatter3d(x=[x_end], y=[y_end], z=[0.3], mode='markers', marker=dict(color=color, size=7, line=dict(color='white', width=2)), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        scene=dict(xaxis=dict(range=[-300, 300], visible=False), yaxis=dict(range=[-20, 450], visible=False), zaxis=dict(range=[0, 150], visible=False), aspectmode='manual', aspectratio=dict(x=1.3, y=1.3, z=0.3), camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=-1.5, z=0.6))),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def render_3d_strikezone(df):
    fig = go.Figure()
    
    # Photorealistic Home Plate
    plate_x, plate_y = [-0.708, 0.708, 0.708, 0, -0.708, -0.708], [0, 0, 0.708, 1.417, 0.708, 0]
    fig.add_trace(go.Scatter3d(x=plate_x, y=plate_y, z=np.zeros(6), mode='lines', surfaceaxis=2, surfacecolor='white', line=dict(color='#0F172A', width=6), hoverinfo='skip', showlegend=False))
    
    # Volumetric Glass Zone Box
    sz_z_lines, sz_x_lines = [1.5, 2.166, 2.833, 3.5], [-0.833, -0.277, 0.277, 0.833]
    for y_plane in [0, 1.417]:
        fig.add_trace(go.Scatter3d(x=[-0.833, 0.833, 0.833, -0.833, -0.833], y=np.full(5, y_plane), z=[1.5, 1.5, 3.5, 3.5, 1.5], mode='lines', line=dict(color='rgba(215, 25, 32, 0.9)' if y_plane==0 else 'rgba(148, 163, 184, 0.5)', width=5), hoverinfo='skip', showlegend=False))
        for z in sz_z_lines[1:3]: fig.add_trace(go.Scatter3d(x=[-0.833, 0.833], y=[y_plane, y_plane], z=[z, z], mode='lines', line=dict(color='rgba(148, 163, 184, 0.3)', width=2), hoverinfo='skip', showlegend=False))
        for x in sz_x_lines[1:3]: fig.add_trace(go.Scatter3d(x=[x, x], y=[y_plane, y_plane], z=[1.5, 3.5], mode='lines', line=dict(color='rgba(148, 163, 184, 0.3)', width=2), hoverinfo='skip', showlegend=False))
    
    # Connectors to create 3D Depth
    for x in [-0.833, 0.833]:
        for z in [1.5, 3.5]: fig.add_trace(go.Scatter3d(x=[x, x], y=[0, 1.417], z=[z, z], mode='lines', line=dict(color='rgba(148, 163, 184, 0.5)', width=4), hoverinfo='skip', showlegend=False))

    if not df.empty:
        df_cp = df.copy()
        # Fallback mappings for robust Contact Depth Geometry
        df_cp['Depth'] = df_cp['ContactPositionY'] if 'ContactPositionY' in df_cp.columns and not df_cp['ContactPositionY'].isna().all() else 0.5
        df_cp['Height'] = df_cp['ContactPositionZ'] if 'ContactPositionZ' in df_cp.columns and not df_cp['ContactPositionZ'].isna().all() else df_cp['PlateLocHeight']
        df_cp['Side'] = df_cp['ContactPositionX'] if 'ContactPositionX' in df_cp.columns and not df_cp['ContactPositionX'].isna().all() else df_cp['PlateLocSide']
        
        valid_cp = df_cp.dropna(subset=['Side', 'Height', 'Depth', 'ExitSpeed'])
        if not valid_cp.empty:
            fig.add_trace(go.Scatter3d(
                x=valid_cp['Side'], y=valid_cp['Depth'], z=valid_cp['Height'], mode='markers',
                marker=dict(size=12, color=valid_cp['ExitSpeed'], colorscale='Turbo', showscale=True, colorbar=dict(title="EV (mph)", len=0.8, thickness=20), line=dict(color='black', width=1.5)),
                hovertemplate="<b style='color:#0F172A; font-size:14px;'>EV: %{marker.color:.1f} mph</b><br>Side: %{x:.2f} ft<br>Depth: %{y:.2f} ft<br>Height: %{z:.2f} ft<extra></extra>",
                name='Contact Point'
            ))
            # Drop shadow indicator on the ground
            fig.add_trace(go.Scatter3d(x=valid_cp['Side'], y=valid_cp['Depth'], z=np.zeros(len(valid_cp)), mode='markers', marker=dict(color='rgba(0,0,0,0.3)', size=6), hoverinfo='skip', showlegend=False))

    fig.update_layout(
        scene=dict(xaxis=dict(title='Side to Side (ft)', range=[-3, 3], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True), yaxis=dict(title='Depth: Catcher ⬅️ Pitcher (ft)', range=[-1, 4], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True), zaxis=dict(title='Height (ft)', range=[0, 5], gridcolor='#E2E8F0', backgroundcolor='#F8FAFC', showbackground=True), aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.9), camera=dict(eye=dict(x=1.4, y=-1.6, z=0.8))),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_wedge_polygon(r_inner, r_outer, theta1, theta2, num_points=40):
    t1_rad, t2_rad = np.radians(theta1), np.radians(theta2)
    theta_vals = np.linspace(t1_rad, t2_rad, num_points)
    x_outer = r_outer * np.sin(theta_vals)
    y_outer = r_outer * np.cos(theta_vals)
    x_inner = r_inner * np.sin(theta_vals[::-1])
    y_inner = r_inner * np.cos(theta_vals[::-1])
    return np.concatenate([x_outer, x_inner, [x_outer[0]]]), np.concatenate([y_outer, y_inner, [y_outer[0]]])

def render_2d_field_heatmap(df, metric='ExitSpeed'):
    fig = go.Figure()
    
    theta_arc = np.linspace(-np.pi/4, np.pi/4, 100)
    fig.add_trace(go.Scatter(x=400 * np.sin(theta_arc), y=400 * np.cos(theta_arc), mode='lines', line=dict(color='#64748B', width=2), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 400*np.sin(np.pi/4)], y=[0, 400*np.cos(np.pi/4)], mode='lines', line=dict(color='#64748B', width=2), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 400*np.sin(-np.pi/4)], y=[0, 400*np.cos(-np.pi/4)], mode='lines', line=dict(color='#64748B', width=2), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=180 * np.sin(theta_arc), y=180 * np.cos(theta_arc), mode='lines', line=dict(color='#94A3B8', width=2, dash='dash'), hoverinfo='skip', showlegend=False))

    zones = {
        'IF_LL': {'r': (0, 180), 't': (-45, -22.5)}, 'IF_LC': {'r': (0, 180), 't': (-22.5, 0)},
        'IF_RC': {'r': (0, 180), 't': (0, 22.5)}, 'IF_RR': {'r': (0, 180), 't': (22.5, 45)},
        'OF_L': {'r': (180, 400), 't': (-45, -15)}, 'OF_C': {'r': (180, 400), 't': (-15, 15)}, 'OF_R': {'r': (180, 400), 't': (15, 45)}
    }

    df_copy = df.copy()
    df_copy['FieldZone'] = df_copy.apply(lambda r: assign_field_zone(r['Distance'], r['Direction']), axis=1)
    avgs = df_copy.groupby('FieldZone')[metric].mean()
    
    vmin, vmax = (75, 100) if metric == 'ExitSpeed' else (0, 35)
    cmap = plt.get_cmap('coolwarm' if metric == 'ExitSpeed' else 'viridis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for z, params in zones.items():
        val = avgs.get(z, np.nan)
        color = 'rgba(241, 245, 249, 0.4)' 
        text_val = "N/A"
        if not pd.isna(val):
            rgba = cmap(norm(val))
            color = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.9)'
            text_val = f"{val:.1f}{' mph' if metric == 'ExitSpeed' else '°'}"

        x_pts, y_pts = create_wedge_polygon(params['r'][0], params['r'][1], params['t'][0], params['t'][1])
        fig.add_trace(go.Scatter(x=x_pts, y=y_pts, fill='toself', fillcolor=color, mode='lines', line=dict(color='white', width=1.5), hoverinfo='skip', showlegend=False))
        
        mid_r = params['r'][0] + (params['r'][1] - params['r'][0])/2
        mid_t = np.radians(params['t'][0] + (params['t'][1] - params['t'][0])/2)
        fig.add_trace(go.Scatter(x=[mid_r * np.sin(mid_t)], y=[mid_r * np.cos(mid_t)], mode='text', text=[text_val], textfont=dict(color='black', size=15, family='Inter', weight='900'), hoverinfo='skip', showlegend=False))

    fig.update_layout(xaxis=dict(visible=False, range=[-300, 300]), yaxis=dict(visible=False, range=[-20, 420], scaleanchor='x', scaleratio=1), margin=dict(l=0, r=0, b=0, t=40), title=dict(text=f"AVERAGE {metric.upper() if metric == 'ExitSpeed' else 'LAUNCH ANGLE'} BY FIELD QUADRANT", font=dict(size=14, family='Inter', weight='bold', color='#0F172A'), x=0.5, y=0.95), paper_bgcolor='white', plot_bgcolor='white')
    return fig

def render_2d_sz_heatmap(df, metric='ExitSpeed'):
    fig = go.Figure()
    
    plate_x, plate_y = [-0.708, 0.708, 0.708, 0, -0.708, -0.708], [0, 0, 0.25, 0.5, 0.25, 0]
    fig.add_trace(go.Scatter(x=plate_x, y=plate_y, mode='lines', fill='toself', fillcolor='white', line=dict(color='#0F172A', width=2), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[-0.833, 0.833, 0.833, -0.833, -0.833], y=[1.5, 1.5, 3.5, 3.5, 1.5], mode='lines', line=dict(color='#0F172A', width=3), hoverinfo='skip', showlegend=False))

    sz_rects = {
        'Z1': {'x': -0.833, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z2': {'x': -0.277, 'y': 2.833, 'w': 0.554, 'h': 0.667}, 'Z3': {'x': 0.277, 'y': 2.833, 'w': 0.556, 'h': 0.667},
        'Z4': {'x': -0.833, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z5': {'x': -0.277, 'y': 2.166, 'w': 0.554, 'h': 0.667}, 'Z6': {'x': 0.277, 'y': 2.166, 'w': 0.556, 'h': 0.667},
        'Z7': {'x': -0.833, 'y': 1.5, 'w': 0.556, 'h': 0.666}, 'Z8': {'x': -0.277, 'y': 1.5, 'w': 0.554, 'h': 0.666}, 'Z9': {'x': 0.277, 'y': 1.5, 'w': 0.556, 'h': 0.666},
        'S1': {'x': -1.166, 'y': 3.5, 'w': 2.332, 'h': 0.333}, 'S3': {'x': -1.166, 'y': 1.167, 'w': 2.332, 'h': 0.333},
        'S2': {'x': -1.166, 'y': 1.5, 'w': 0.333, 'h': 2.0}, 'S4': {'x': 0.833, 'y': 1.5, 'w': 0.333, 'h': 2.0}
    }

    df_copy = df.copy()
    df_copy['SZ'] = df_copy.apply(lambda r: assign_sz_zone(r['PlateLocSide'], r['PlateLocHeight']), axis=1)
    avgs = df_copy.groupby('SZ')[metric].mean()
    
    vmin, vmax = (75, 100) if metric == 'ExitSpeed' else (0, 35)
    cmap = plt.get_cmap('coolwarm' if metric == 'ExitSpeed' else 'viridis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for z, r in sz_rects.items():
        val = avgs.get(z, np.nan)
        color = 'rgba(241, 245, 249, 0.4)'
        text_val = "N/A"
        if not pd.isna(val):
            rgba = cmap(norm(val))
            color = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.95)'
            text_val = f"{val:.1f}"

        x_pts = [r['x'], r['x']+r['w'], r['x']+r['w'], r['x'], r['x']]
        y_pts = [r['y'], r['y'], r['y']+r['h'], r['y']+r['h'], r['y']]
        line_color = 'white' if z.startswith('Z') else 'rgba(0,0,0,0)'
        fig.add_trace(go.Scatter(x=x_pts, y=y_pts, fill='toself', fillcolor=color, mode='lines', line=dict(color=line_color, width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[r['x']+r['w']/2], y=[r['y']+r['h']/2], mode='text', text=[text_val], textfont=dict(color='black', size=16 if z.startswith('Z') else 11, family='Inter', weight='900'), hoverinfo='skip', showlegend=False))

    fig.update_layout(xaxis=dict(visible=False, range=[-2, 2]), yaxis=dict(visible=False, range=[0, 4.5], scaleanchor='x', scaleratio=1), margin=dict(l=0, r=0, b=0, t=40), title=dict(text=f"AVERAGE {metric.upper() if metric == 'ExitSpeed' else 'LAUNCH ANGLE'} BY PITCH LOCATION", font=dict(size=14, family='Inter', weight='bold', color='#0F172A'), x=0.5, y=0.95), paper_bgcolor='white', plot_bgcolor='white')
    return fig

# ==========================================
# 4. MATPLOTLIB PDF EXPORT ENGINE
# ==========================================
def generate_pdf_buffer(df, kpis, player_name, side, date_str):
    fig = plt.figure(figsize=(14, 11)) 
    fig.patches.append(Rectangle((0.015, 0.015), 0.97, 0.97, fill=False, edgecolor='#0F172A', lw=2, transform=fig.transFigure))
    fig.text(0.04, 0.94, f"{player_name.upper()}", fontsize=28, fontweight='900', color='#0F172A')
    fig.text(0.35, 0.94, f"|  {side}", fontsize=26, fontweight='300', color='#D71920')
    fig.text(0.04, 0.91, f"OMAHA EXECUTIVE ANALYTICS • CONTEXT: {date_str}", fontsize=12, color='#64748B', fontweight='bold')
    fig.add_artist(plt.Line2D((0.04, 0.89), (0.89, 0.89), color='#E2E8F0', linewidth=2))

    box_width = 0.085 
    x_offsets = np.linspace(0.025, 0.885, 9).tolist()
    for idx, (label, val) in enumerate(kpis):
        fig.patches.append(FancyBboxPatch((x_offsets[idx], 0.80), box_width, 0.065, boxstyle="round,pad=0.01", fc="#F8FAFC", ec="#CBD5E1", lw=1.5, transform=fig.transFigure))
        fig.text(x_offsets[idx] + (box_width / 2), 0.845, label, fontsize=8, fontweight='bold', color='#64748B', ha='center')
        fig.text(x_offsets[idx] + (box_width / 2), 0.815, str(val), fontsize=14, fontweight='900', color='#0F172A', ha='center')

    def _draw_field(ax):
        angles = np.linspace(-45, 45, 100)
        ax.plot(380 * np.sin(np.radians(angles)), 380 * np.cos(np.radians(angles)), color='#475569', lw=2, zorder=3)
        ax.add_patch(Wedge(center=(0, 0), r=380, theta1=45, theta2=135, facecolor='#F8FAFC', edgecolor='black', lw=1, zorder=1))
        ax.add_patch(Wedge(center=(0, 0), r=154, theta1=45, theta2=135, facecolor='#F1F5F9', edgecolor='black', lw=1, zorder=2))
        for a in [-22.5, 0, 22.5]: ax.plot([0, 380 * np.sin(np.radians(a))], [0, 380 * np.cos(np.radians(a))], color='#CBD5E1', lw=1, zorder=3)
        ax.plot([0, 380 * np.sin(np.radians(-45))], [0, 380 * np.cos(np.radians(-45))], color='#475569', lw=2, zorder=3)
        ax.plot([0, 380 * np.sin(np.radians(45))], [0, 380 * np.cos(np.radians(45))], color='#475569', lw=2, zorder=3)

    ax1 = fig.add_axes([0.025, 0.42, 0.28, 0.34])
    _draw_field(ax1)
    ax1.scatter(df['Distance'] * np.sin(np.radians(df['Direction'])), df['Distance'] * np.cos(np.radians(df['Direction'])), c=df['ExitSpeed'], cmap='coolwarm', vmin=75, vmax=105, s=40, edgecolors='black', lw=0.5, zorder=6)
    ax1.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Batted Ball Spray (Color = EV)")

    df_copy = df.copy()
    df_copy['Zone'] = df_copy.apply(lambda r: assign_field_zone(r['Distance'], r['Direction']), axis=1)
    avgs, avgs_la = df_copy.groupby('Zone')['ExitSpeed'].mean(), df_copy.groupby('Zone')['Angle'].mean()
    norm, cmap = mcolors.Normalize(vmin=80, vmax=100), plt.get_cmap('coolwarm')
    norm_la, cmap_la = mcolors.Normalize(vmin=0, vmax=35), plt.get_cmap('viridis')

    ZONES_CONFIG = { 'IF_LL': {'r_inner': 0, 'width': 180, 'theta1': 112.5, 'theta2': 135, 'text_r': 160, 'text_theta': 123.75}, 'IF_LC': {'r_inner': 0, 'width': 180, 'theta1': 90, 'theta2': 112.5, 'text_r': 160, 'text_theta': 101.25}, 'IF_RC': {'r_inner': 0, 'width': 180, 'theta1': 67.5, 'theta2': 90, 'text_r': 160, 'text_theta': 78.75}, 'IF_RR': {'r_inner': 0, 'width': 180, 'theta1': 45, 'theta2': 67.5, 'text_r': 160, 'text_theta': 56.25}, 'OF_L': {'r_inner': 180, 'width': 200, 'theta1': 105, 'theta2': 135, 'text_r': 350, 'text_theta': 120}, 'OF_C': {'r_inner': 180, 'width': 200, 'theta1': 75, 'theta2': 105, 'text_r': 350, 'text_theta': 90}, 'OF_R': {'r_inner': 180, 'width': 200, 'theta1': 45, 'theta2': 75, 'text_r': 350, 'text_theta': 60} }
    SZ_ZONES_CONFIG = { 'Z1': {'x': -0.833, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z2': {'x': -0.277, 'y': 2.833, 'w': 0.554, 'h': 0.667}, 'Z3': {'x': 0.277, 'y': 2.833, 'w': 0.556, 'h': 0.667}, 'Z4': {'x': -0.833, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z5': {'x': -0.277, 'y': 2.166, 'w': 0.554, 'h': 0.667}, 'Z6': {'x': 0.277, 'y': 2.166, 'w': 0.556, 'h': 0.667}, 'Z7': {'x': -0.833, 'y': 1.5, 'w': 0.556, 'h': 0.666}, 'Z8': {'x': -0.277, 'y': 1.5, 'w': 0.554, 'h': 0.666}, 'Z9': {'x': 0.277, 'y': 1.5, 'w': 0.556, 'h': 0.666}, 'S1': {'x': -1.166, 'y': 3.5, 'w': 2.332, 'h': 0.333}, 'S3': {'x': -1.166, 'y': 1.167, 'w': 2.332, 'h': 0.333}, 'S2': {'x': -1.166, 'y': 1.5, 'w': 0.333, 'h': 2.0}, 'S4': {'x': 0.833, 'y': 1.5, 'w': 0.333, 'h': 2.0} }

    ax2 = fig.add_axes([0.345, 0.42, 0.28, 0.34]); _draw_field(ax2)
    ax3 = fig.add_axes([0.665, 0.42, 0.28, 0.34]); _draw_field(ax3)
    
    for z, c in ZONES_CONFIG.items():
        v_ev, v_la = avgs.get(z, np.nan), avgs_la.get(z, np.nan)
        ax2.add_patch(Wedge(center=(0, 0), r=c['r_inner']+c['width'], theta1=c['theta1'], theta2=c['theta2'], width=c['width'], facecolor=cmap(norm(v_ev)) if not pd.isna(v_ev) else '#FFFFFF', edgecolor='#CBD5E1', alpha=0.9, zorder=1))
        ax3.add_patch(Wedge(center=(0, 0), r=c['r_inner']+c['width'], theta1=c['theta1'], theta2=c['theta2'], width=c['width'], facecolor=cmap_la(norm_la(v_la)) if not pd.isna(v_la) else '#FFFFFF', edgecolor='#CBD5E1', alpha=0.9, zorder=1))
        if not pd.isna(v_ev): ax2.text(c['text_r']*np.cos(np.radians(c['text_theta'])), c['text_r']*np.sin(np.radians(c['text_theta'])), f"{v_ev:.1f}", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2), zorder=4)
        if not pd.isna(v_la): ax3.text(c['text_r']*np.cos(np.radians(c['text_theta'])), c['text_r']*np.sin(np.radians(c['text_theta'])), f"{v_la:.1f}°", ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2), zorder=4)
    ax2.set(xlim=(-300, 300), ylim=(-30, 410), aspect='equal', xticks=[], yticks=[], title="Avg EV by Quadrant")
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

    ax4 = fig.add_axes([0.025, 0.06, 0.28, 0.34]); _draw_sz(ax4, 'ExitSpeed', 80, 100, 'coolwarm', "Avg EV by Pitch Location")
    ax5 = fig.add_axes([0.345, 0.06, 0.28, 0.34]); _draw_sz(ax5, 'Angle', 0, 35, 'viridis', "Avg LA by Pitch Location", is_ev=False)

    ax6 = fig.add_axes([0.665, 0.06, 0.28, 0.34])
    dep_col = 'ContactPositionZ' if 'ContactPositionZ' in df.columns and not df['ContactPositionZ'].isna().all() else 'ContactPositionY'
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
# 5. DASH UI LAYOUT
# ==========================================
SIDEBAR_STYLE = { "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "22rem", "padding": "2rem 1.5rem", "background-color": "#0F172A", "color": "white", "overflowY": "auto"}
CONTENT_STYLE = { "margin-left": "23rem", "padding": "2rem", "background-color": "#F8FAFC", "min-height": "100vh"}

sidebar = html.Div([
    html.Img(src="https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Omaha_Mavericks_logo.svg/1200px-Omaha_Mavericks_logo.svg.png", style={"width": "140px", "display": "block", "margin": "0 auto 30px auto"}),
    
    html.H5("Performance Filters", style={"color": "#94A3B8", "fontSize": "13px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "15px", "fontWeight": "bold"}),
    
    html.Label("Hitter Profile", style={"fontSize": "12px", "fontWeight": "600"}),
    dcc.Dropdown(id='hitter-dropdown', style={"color": "black", "marginBottom": "15px"}),
    
    html.Label("Date Range", style={"fontSize": "12px", "fontWeight": "600"}),
    dcc.Dropdown(id='date-dropdown', style={"color": "black", "marginBottom": "15px"}),
    
    dbc.Accordion([
        dbc.AccordionItem([
            html.Label("Session Filters", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-session", options=[], multi=True, style={"color": "black", "marginBottom": "10px"}),
            html.Label("Season Filters", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-season", options=[], multi=True, style={"color": "black", "marginBottom": "10px"}),
            html.Label("Opponent", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-opponent", options=[], multi=True, style={"color": "black", "marginBottom": "10px"}),
            html.Label("Pitcher Hand", style={"fontSize": "12px"}),
            dbc.Checklist(id="filter-hand", options=[{"label": "Vs. RHP", "value": "Right"}, {"label": "Vs. LHP", "value": "Left"}], inline=True, style={"fontSize": "12px", "marginBottom": "10px"}),
            html.Label("Pitch Type", style={"fontSize": "12px"}),
            dbc.Checklist(id="filter-pitch", options=[], inline=True, style={"fontSize": "12px", "marginBottom": "10px"}),
        ], title="Situational Context", style={"backgroundColor": "#1E293B", "border": "none", "color": "white"}),
        dbc.AccordionItem([
            html.Label("Play Result", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-result", options=[], multi=True, style={"color": "black", "marginBottom": "10px"}),
            html.Label("Pitch Call", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-call", options=[], multi=True, style={"color": "black", "marginBottom": "10px"}),
            html.Label("Count", style={"fontSize": "12px"}),
            dcc.Dropdown(id="filter-count", options=[{"label": f"{b}-{s}", "value": f"{b}-{s}"} for b in range(4) for s in range(3)], multi=True, style={"color": "black", "marginBottom": "10px"}),
        ], title="Results & Counts", style={"backgroundColor": "#1E293B", "border": "none", "color": "white"})
    ], start_collapsed=True, style={"marginBottom": "20px"}),
], style=SIDEBAR_STYLE)

main_content = html.Div([
    html.Div(id="dashboard-header", style={"background": "linear-gradient(135deg, #0B0F19 0%, #1A2235 100%)", "borderRadius": "16px", "padding": "30px 40px", "color": "white", "marginBottom": "30px", "borderLeft": "8px solid #D71920", "boxShadow": "0 20px 25px -5px rgba(0, 0, 0, 0.1)"}),
    html.Div(id="kpi-grid", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))", "gap": "15px", "marginBottom": "30px"}),
    
    dbc.Tabs([
        dbc.Tab(html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='3d-stadium-graph', style={"height": "750px"}), width=12)])
        ], style={"padding": "20px"}), label="🌐 Interactive Spray Chart", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"}),
        
        dbc.Tab(html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='3d-strikezone-graph', style={"height": "750px"}), width=12)])
        ], style={"padding": "20px"}), label="⚾ 3D Strike Zone", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"}),
        
        dbc.Tab(html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id='2d-ev-field', style={"height": "600px"}), width=6),
                dbc.Col(dcc.Graph(id='2d-la-field', style={"height": "600px"}), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='2d-ev-sz', style={"height": "600px"}), width=6),
                dbc.Col(dcc.Graph(id='2d-la-sz', style={"height": "600px"}), width=6)
            ], style={"marginTop": "20px"})
        ], style={"padding": "20px"}), label="🔥 Heatmap Matrices", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"}),
        
        dbc.Tab(html.Div([
            html.H4("Batting Line Summary", style={"marginBottom": "20px", "fontWeight": "bold"}),
            html.Div(id="batting-line-table", style={"marginBottom": "40px"}),
            html.H4("Raw Event Log", style={"marginBottom": "20px", "fontWeight": "bold"}),
            html.Div(id="data-table-container")
        ], style={"padding": "30px", "backgroundColor": "white", "borderRadius": "8px", "border": "1px solid #E2E8F0"}), label="📊 Performance Tables", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"}),
        
        dbc.Tab(html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Executive PDF Report", style={"marginBottom": "10px", "fontWeight": "bold"}),
                    html.P("Generate a high-resolution, 6-chart vector PDF report designed for print or iPad review.", style={"color": "#64748B"}),
                    dbc.Button("Generate Professional PDF", id="btn-pdf", color="dark", className="mb-4", style={"width": "100%", "fontWeight": "bold"}),
                    dcc.Download(id="download-pdf")
                ], width=6)
            ])
        ], style={"padding": "30px"}), label="📄 Reporting", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"}),

        dbc.Tab(html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Ingest New Trackman Data", style={"fontWeight": "bold", "marginBottom": "20px"}),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Select(id="upload-session-type", options=[{"label": "BP Session", "value": "BP"}, {"label": "Game", "value": "Game"}, {"label": "Scrimmage", "value": "Scrimmage"}], placeholder="Select Session Type...", style={"marginBottom": "15px"}),
                            dbc.Input(id="upload-season", placeholder="Season (e.g., 2026 Spring)...", style={"marginBottom": "15px"}),
                            dbc.Input(id="upload-opponent", placeholder="Opponent Name / Session Info...", style={"marginBottom": "20px"}),
                            dcc.Upload(id='upload-data', children=html.Div(['Drop CSV or ', html.A('Select File', style={"fontWeight": "bold"})]), style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderColor': '#D71920', 'borderRadius': '8px', 'textAlign': 'center', 'cursor': 'pointer', 'backgroundColor': '#f8fafc'}),
                            html.Div(id='upload-output', style={"color": "#16a34a", "fontSize": "14px", "marginTop": "15px", "fontWeight": "bold"})
                        ])
                    ], style={"border": "1px solid #E2E8F0", "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.05)"})
                ], width=5),
                dbc.Col([
                    html.H4("Manage Uploaded Sessions", style={"fontWeight": "bold", "marginBottom": "20px"}),
                    html.Div(id="manage-sessions-table", style={"backgroundColor": "white", "border": "1px solid #E2E8F0", "borderRadius": "8px", "padding": "15px", "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.05)"}),
                    html.Div([
                        dbc.Button("Delete Selected Session", id="btn-delete-session", color="danger", style={"marginTop": "15px", "fontWeight": "bold"}),
                        html.Div(id="delete-output", style={"color": "#D71920", "marginTop": "10px", "fontWeight": "bold"})
                    ])
                ], width=7)
            ])
        ], style={"padding": "30px"}), label="⚙️ Data Hub", tab_style={"backgroundColor": "#E2E8F0", "border": "none"}, active_tab_style={"backgroundColor": "white", "color": "#0F172A", "fontWeight": "bold", "borderTop": "3px solid #D71920"})
    ])
], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, main_content])

# ==========================================
# 6. REACTIVE CALLBACKS
# ==========================================

@app.callback(
    Output('upload-output', 'children'),
    Input('upload-data', 'contents'),
    [State('upload-session-type', 'value'), State('upload-season', 'value'), State('upload-opponent', 'value')],
    prevent_initial_call=True
)
def process_upload(contents, session_type, season, opponent):
    if contents is None: return ""
    if not session_type: return "Error: Please select a Session Type before uploading."
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Dual-decoder to seamlessly handle Trackman's UTF-8-BOM files
        try:
            df_upload = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        except:
            df_upload = pd.read_csv(io.StringIO(decoded.decode('latin1')))
        
        cols_to_keep = [
            'PitchNo', 'Date', 'Time', 'Pitcher', 'Batter', 'BatterSide', 'PitcherThrows',
            'TaggedPitchType', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'HitSpinRate', 
            'HangTime', 'PlateLocHeight', 'PlateLocSide', 'ContactPositionX', 
            'ContactPositionY', 'ContactPositionZ', 'PitchCall', 'PlayResult', 'TaggedHitType',
            'KorBB', 'Balls', 'Strikes', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle'
        ]
        available_cols = [c for c in cols_to_keep if c in df_upload.columns]
        df_cleaned = df_upload[available_cols].copy()
        
        # Force critical batted ball metrics to be pure floats 
        numeric_cols = ['ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocHeight', 'PlateLocSide', 'ContactPositionX', 'ContactPositionY', 'ContactPositionZ']
        for col in numeric_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        df_cleaned['SessionType'] = session_type
        df_cleaned['Season'] = season if season else "Unspecified"
        df_cleaned['Opponent'] = opponent if opponent else "Unspecified"
        
        conn = sqlite3.connect(DB_NAME)
        existing_dates = pd.read_sql("SELECT DISTINCT Date FROM trackman_data", conn)['Date'].tolist()
        if not df_cleaned.empty and str(df_cleaned['Date'].iloc[0]) in existing_dates:
            conn.close()
            return "Session already exists in database."

        df_cleaned.to_sql('trackman_data', conn, if_exists='append', index=False)
        conn.close()
        return f"✅ Data ingested successfully ({session_type})."
    except Exception as e:
        return f"Error processing file: {str(e)}"

@app.callback(
    Output('manage-sessions-table', 'children'),
    [Input('upload-output', 'children'), Input('delete-output', 'children')]
)
def update_manage_table(upload_msg, delete_msg):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql("SELECT Date, SessionType, Season, Opponent, COUNT(*) as Pitches FROM trackman_data GROUP BY Date, SessionType, Season, Opponent ORDER BY Date DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty: return "No sessions found in the database."
    
    return dash_table.DataTable(
        id='session-delete-table',
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        row_selectable="single",
        page_size=8,
        style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Inter'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8FAFC'}]
    )

@app.callback(
    Output('delete-output', 'children'),
    Input('btn-delete-session', 'n_clicks'),
    State('session-delete-table', 'selected_rows'),
    State('session-delete-table', 'data'),
    prevent_initial_call=True
)
def delete_session(n_clicks, selected_rows, table_data):
    if not selected_rows: return "Please select a session to delete."
    row = table_data[selected_rows[0]]
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM trackman_data WHERE Date=? AND SessionType=? AND Opponent=?", (row['Date'], row['SessionType'], row['Opponent']))
    conn.commit()
    conn.close()
    return f"🗑️ Deleted {row['SessionType']} session on {row['Date']}."

@app.callback(
    [Output('hitter-dropdown', 'options'), Output('hitter-dropdown', 'value')],
    [Input('upload-output', 'children'), Input('delete-output', 'children')]
)
def update_hitters(_1, _2):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT DISTINCT Batter FROM trackman_data", conn)
    conn.close()
    if df.empty: return [], None
    batters = sorted(df['Batter'].dropna().tolist())
    return [{'label': b, 'value': b} for b in batters], batters[0]

@app.callback(
    [Output('date-dropdown', 'options'), Output('date-dropdown', 'value'),
     Output('filter-session', 'options'), Output('filter-season', 'options'), Output('filter-opponent', 'options'),
     Output('filter-pitch', 'options'), Output('filter-result', 'options'),
     Output('filter-call', 'options')],
    Input('hitter-dropdown', 'value')
)
def update_filters(hitter):
    if not hitter: return [], None, [], [], [], [], [], []
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f"SELECT DISTINCT Date, SessionType, Season, Opponent, TaggedPitchType, PlayResult, PitchCall FROM trackman_data WHERE Batter = '{hitter}'", conn)
    conn.close()
    
    dates = ["All-Time"] + sorted(df['Date'].dropna().unique().tolist(), reverse=True)
    sessions = [{'label': s, 'value': s} for s in sorted(df['SessionType'].dropna().unique().tolist())]
    seasons = [{'label': s, 'value': s} for s in sorted(df['Season'].dropna().unique().tolist())]
    opponents = [{'label': o, 'value': o} for o in sorted(df['Opponent'].dropna().unique().tolist())]
    pitches = [{'label': p, 'value': p} for p in sorted(df['TaggedPitchType'].dropna().unique().tolist())]
    results = [{'label': r, 'value': r} for r in sorted(df['PlayResult'].dropna().unique().tolist())]
    calls = [{'label': c, 'value': c} for c in sorted(df['PitchCall'].dropna().unique().tolist())]
    
    return [{'label': d, 'value': d} for d in dates], "All-Time", sessions, seasons, opponents, pitches, results, calls

@app.callback(
    [Output('dashboard-header', 'children'), Output('kpi-grid', 'children'), 
     Output('3d-stadium-graph', 'figure'), Output('3d-strikezone-graph', 'figure'),
     Output('2d-ev-field', 'figure'), Output('2d-la-field', 'figure'),
     Output('2d-ev-sz', 'figure'), Output('2d-la-sz', 'figure'),
     Output('batting-line-table', 'children'), Output('data-table-container', 'children')],
    [Input('hitter-dropdown', 'value'), Input('date-dropdown', 'value'),
     Input('filter-session', 'value'), Input('filter-season', 'value'), Input('filter-opponent', 'value'),
     Input('filter-hand', 'value'), Input('filter-pitch', 'value'),
     Input('filter-result', 'value'), Input('filter-call', 'value'),
     Input('filter-count', 'value')]
)
def update_dashboard(hitter, date, sessions, seasons, opponents, hands, pitches, results, calls, counts):
    if not hitter: return dash.no_update, dash.no_update, go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), "", ""
    
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM trackman_data WHERE Batter = '{hitter}'"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if date and date != "All-Time": df = df[df['Date'] == date]
    if sessions: df = df[df['SessionType'].isin(sessions)]
    if seasons: df = df[df['Season'].isin(seasons)]
    if opponents: df = df[df['Opponent'].isin(opponents)]
    if hands: df = df[df['PitcherThrows'].isin(hands)]
    if pitches: df = df[df['TaggedPitchType'].isin(pitches)]
    if results: df = df[df['PlayResult'].isin(results)]
    if calls: df = df[df['PitchCall'].isin(calls)]
    if counts:
        df['CountStr'] = df['Balls'].astype(str) + "-" + df['Strikes'].astype(str)
        df = df[df['CountStr'].isin(counts)]

    batted_balls = df.dropna(subset=['ExitSpeed', 'Angle', 'Direction', 'Distance'])
    
    if batted_balls.empty:
        empty_header = html.Div([html.H1(hitter.upper(), style={"fontSize": "42px", "fontWeight": "900", "margin": 0}), html.Div("NO BATTED BALL EVENTS FOUND FOR CURRENT FILTERS", style={"fontSize": "15px", "fontWeight": "600", "color": "#D71920"})])
        return empty_header, "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), "", ""

    side = get_batter_side(df['BatterSide'].iloc[0] if 'BatterSide' in df.columns else "UNK")
    total_swings = len(df[df['PitchCall'].isin(['InPlay', 'Foul', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable'])])
    bip = len(batted_balls)
    avg_ev, max_ev = batted_balls['ExitSpeed'].mean(), batted_balls['ExitSpeed'].max()
    ev90 = np.percentile(batted_balls['ExitSpeed'], 90) 
    hh_pct = (len(batted_balls[batted_balls['ExitSpeed'] >= 95]) / bip) * 100 if bip > 0 else 0
    swsp_pct = (len(batted_balls[(batted_balls['Angle'] >= 8) & (batted_balls['Angle'] <= 32)]) / bip) * 100 if bip > 0 else 0
    avg_la, std_la = batted_balls['Angle'].mean(), batted_balls['Angle'].std()
    avg_dist = batted_balls['Distance'].mean()

    header = [
        html.Div([
            html.H1(hitter.upper(), style={"fontSize": "42px", "fontWeight": "900", "margin": 0, "letterSpacing": "-1px"}),
            html.Div(f"BATS: {side}  |  RANGE: {date}  |  BATTED BALL EVENTS: {bip}", style={"fontSize": "15px", "fontWeight": "600", "color": "#94A3B8", "letterSpacing": "1px", "marginTop": "6px"})
        ])
    ]

    kpis = [("TOTAL SWINGS", total_swings), ("HARD HIT %", f"{hh_pct:.1f}%"), ("SWEET SPOT %", f"{swsp_pct:.1f}%"), ("AVG EV", f"{avg_ev:.1f}"), ("90th% EV", f"{ev90:.1f}"), ("MAX EV", f"{max_ev:.1f}"), ("AVG LA", f"{avg_la:.1f}°"), ("LA DEV (SD)", f"{std_la:.1f}°"), ("AVG DIST", f"{avg_dist:.0f} ft")]
    
    kpi_cards = []
    for label, val in kpis:
        val_color = "#D71920" if label in ["90th% EV", "SWEET SPOT %", "HARD HIT %"] else "#0F172A"
        kpi_cards.append(html.Div([
            html.Div(label, style={"fontSize": "11px", "fontWeight": "800", "color": "#64748B", "marginBottom": "5px"}),
            html.Div(val, style={"fontSize": "26px", "fontWeight": "900", "color": val_color})
        ], style={"background": "#FFFFFF", "border": "1px solid #E2E8F0", "borderRadius": "12px", "padding": "20px 10px", "textAlign": "center", "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.03)"}))

    fig_field = render_3d_stadium(batted_balls)
    fig_zone = render_3d_strikezone(batted_balls)
    fig_ev_field = render_2d_field_heatmap(batted_balls, metric='ExitSpeed')
    fig_la_field = render_2d_field_heatmap(batted_balls, metric='Angle')
    fig_ev_sz = render_2d_sz_heatmap(batted_balls, metric='ExitSpeed')
    fig_la_sz = render_2d_sz_heatmap(batted_balls, metric='Angle')

    batting_line_df = get_full_batting_line(df)
    bl_table = dash_table.DataTable(
        data=batting_line_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in batting_line_df.columns],
        style_header={'backgroundColor': '#D71920', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Inter'},
        style_data_conditional=[{'if': {'filter_query': '{Split} = "TOTAL"'}, 'fontWeight': 'bold', 'backgroundColor': '#F1F5F9'}]
    )

    raw_table = dash_table.DataTable(
        data=batted_balls[['PitchNo', 'SessionType', 'Pitcher', 'TaggedPitchType', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlayResult']].sort_values(by='ExitSpeed', ascending=False).to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['PitchNo', 'SessionType', 'Pitcher', 'TaggedPitchType', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlayResult']],
        page_size=15,
        style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Inter'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F8FAFC'}]
    )

    return header, kpi_cards, fig_field, fig_zone, fig_ev_field, fig_la_field, fig_ev_sz, fig_la_sz, bl_table, raw_table

@app.callback(
    Output("download-pdf", "data"),
    Input("btn-pdf", "n_clicks"),
    [State("hitter-dropdown", "value"), State("date-dropdown", "value"),
     State('filter-session', 'value'), State('filter-season', 'value'), State('filter-opponent', 'value')],
    prevent_initial_call=True
)
def download_pdf(n_clicks, hitter, date, sessions, seasons, opponents):
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM trackman_data WHERE Batter = '{hitter}'"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if date and date != "All-Time": df = df[df['Date'] == date]
    if sessions: df = df[df['SessionType'].isin(sessions)]
    if seasons: df = df[df['Season'].isin(seasons)]
    if opponents: df = df[df['Opponent'].isin(opponents)]
    
    batted_balls = df.dropna(subset=['ExitSpeed', 'Angle', 'Direction', 'Distance'])
    side = get_batter_side(df['BatterSide'].iloc[0] if 'BatterSide' in df.columns else "UNK")
    
    swings = len(df[df['PitchCall'].isin(['InPlay', 'Foul', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable'])])
    bip = len(batted_balls)
    kpis = [
        ("SWINGS", swings), ("HARD HIT %", f"{(len(batted_balls[batted_balls['ExitSpeed'] >= 95]) / bip * 100):.1f}%" if bip else "0%"), ("SWEET SPOT %", f"{(len(batted_balls[(batted_balls['Angle'] >= 8) & (batted_balls['Angle'] <= 32)]) / bip * 100):.1f}%" if bip else "0%"),
        ("AVG EV", f"{batted_balls['ExitSpeed'].mean():.1f}"), ("90th% EV", f"{np.percentile(batted_balls['ExitSpeed'], 90):.1f}"), ("MAX EV", f"{batted_balls['ExitSpeed'].max():.1f}"),
        ("AVG LA", f"{batted_balls['Angle'].mean():.1f}°"), ("LA DEV", f"{batted_balls['Angle'].std():.1f}°"), ("AVG DIST", f"{batted_balls['Distance'].mean():.0f} ft")
    ]
    
    date_str = date if date != "All-Time" else "Multiple Sessions Filtered"
    pdf_bytes = generate_pdf_buffer(batted_balls, kpis, hitter, side, date_str)
    return dcc.send_bytes(pdf_bytes, f"Omaha_Dev_Report_{hitter.replace(' ', '_')}.pdf")

if __name__ == '__main__':
    app.run(debug=True)
