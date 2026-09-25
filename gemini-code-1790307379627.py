import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Wedge, FancyBboxPatch
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
        .kpi-label { font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; }
        .kpi-value { font-size: 22px; font-weight: 800; color: #000000; }
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
    
    # Basic check to avoid duplicating exact sessions if uploaded again
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

def assign_sz_zone(x, z):
    if pd.isna(x) or pd.isna(z): return None
    if -0.833 <= x <= 0.833 and 1.5 <= z <= 3.5:
        if z >= 2.833: return 'Z1' if x <= -0.277 else 'Z2' if x <= 0.277 else 'Z3'
        elif z >= 2.166: return 'Z4' if x <= -0.277 else 'Z5' if x <= 0.277 else 'Z6'
        else: return 'Z7' if x <= -0.277 else 'Z8' if x <= 0.277 else 'Z9'
    return 'OOT' # Out of True Zone for simplification

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
            
            # Keep only available columns
            available_cols = [c for c in cols_to_keep if c in df_upload.columns]
            df_cleaned = df_upload[available_cols]
            
            if st.button("Save to Database"):
                success, msg = load_data_to_db(df_cleaned)
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("### API Keys")
    gemini_key = st.text_input("Gemini API Key (For AI Analysis)", type="password")

    st.markdown("---")
    st.markdown("### Dashboard Filters")
    all_data = fetch_data()
    
    if not all_data.empty:
        # Filter: Batter
        batters = sorted(all_data['Batter'].dropna().unique())
        selected_batter = st.selectbox("Select Hitter:", batters)
        
        # Filter: Date / Session
        batter_subset = all_data[all_data['Batter'] == selected_batter]
        dates = ["All-Time"] + sorted(batter_subset['Date'].dropna().unique().tolist(), reverse=True)
        selected_date = st.selectbox("Select Session:", dates)
        
        if selected_date != "All-Time":
            batter_subset = batter_subset[batter_subset['Date'] == selected_date]
            
        # Filter: Batted Balls Only
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
    
    batter_side = get_batter_side(batter_subset['BatterSide'].iloc[0] if 'BatterSide' in batter_subset.columns else "Unk")
    
    st.markdown(f"""
        <div class="header-box">
            <div class="header-title">{selected_batter} <span style="color:#e02424; font-size:22px;">| {batter_side}</span></div>
            <div class="header-sub">OMAHA PLAYER DEVELOPMENT • Session: {selected_date}</div>
        </div>
    """, unsafe_allow_html=True)
    
    kpi_cols = st.columns(7)
    kpis = [
        ("Total Swings", int(total_swings)), ("Balls in Play", int(total_bip)),
        ("Hard Hit (90%)", f"{hh_pct:.1f}%"), ("Avg EV", f"{avg_ev:.1f} mph"),
        ("Max EV", f"{max_ev:.1f} mph"), ("Avg LA", f"{avg_la:.1f}°"), 
        ("Avg Distance", f"{avg_dist:.0f} ft")
    ]
    
    for col, (label, val) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    # --- TABS SETUP ---
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Interactive Visuals", "🔥 2D Heatmaps", "🧠 AI Scouting Analysis", "📄 PDF & Raw Data"])
    
    with tab1:
        st.markdown("### 3D Batted Ball Trajectories")
        # Approximate 3D Trajectory Calculation
        fig_3d = go.Figure()
        
        # Draw field lines
        angles = np.linspace(-45, 45, 100)
        fence_x = 400 * np.sin(np.radians(angles))
        fence_y = 400 * np.cos(np.radians(angles))
        fig_3d.add_trace(go.Scatter3d(x=fence_x, y=fence_y, z=np.zeros_like(fence_x), mode='lines', line=dict(color='black', width=4), name='Fence'))
        fig_3d.add_trace(go.Scatter3d(x=[0, 400*np.sin(np.radians(45))], y=[0, 400*np.cos(np.radians(45))], z=[0,0], mode='lines', line=dict(color='black', width=4), name='Foul Line'))
        fig_3d.add_trace(go.Scatter3d(x=[0, 400*np.sin(np.radians(-45))], y=[0, 400*np.cos(np.radians(-45))], z=[0,0], mode='lines', line=dict(color='black', width=4), name='Foul Line'))
        
        for idx, row in batted_balls.iterrows():
            dist = row['Distance']
            direction = row['Direction']
            angle = row['Angle']
            ev = row['ExitSpeed']
            
            # Simple parabolic approximation
            t = np.linspace(0, 1, 30)
            x_end = dist * np.sin(np.radians(direction))
            y_end = dist * np.cos(np.radians(direction))
            
            x_traj = t * x_end
            y_traj = t * y_end
            # H_max roughly = distance * tan(angle) / 4 (visual scaling)
            h_max = dist * np.tan(np.radians(max(0.1, angle))) * 0.25 
            z_traj = 4 * h_max * t * (1 - t)
            
            color = 'red' if angle > 25 else 'green' if angle >= 10 else 'blue'
            
            fig_3d.add_trace(go.Scatter3d(
                x=x_traj, y=y_traj, z=z_traj,
                mode='lines', line=dict(color=color, width=3),
                hovertemplate=f"<b>EV:</b> {ev:.1f} mph<br><b>LA:</b> {angle:.1f}°<br><b>Dist:</b> {dist:.0f} ft<extra></extra>",
                showlegend=False
            ))
            
            # Landing Point
            fig_3d.add_trace(go.Scatter3d(
                x=[x_end], y=[y_end], z=[0],
                mode='markers', marker=dict(color=color, size=5, symbol='diamond'),
                showlegend=False
            ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='Horizontal (ft)', range=[-300, 300], showgrid=False),
                yaxis=dict(title='Distance (ft)', range=[0, 450], showgrid=False),
                zaxis=dict(title='Height (ft)', range=[0, 150], showgrid=False),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("### 3D Contact Point Matrix")
        if 'ContactPositionZ' in batted_balls.columns and not batted_balls['ContactPositionZ'].isna().all():
            fig_cp = go.Figure()
            fig_cp.add_trace(go.Scatter3d(
                x=batted_balls['PlateLocSide'], 
                y=batted_balls['ContactPositionY'] if 'ContactPositionY' in batted_balls.columns else batted_balls['ContactPositionX'], # Using Y for depth if available, otherwise X proxy
                z=batted_balls['PlateLocHeight'],
                mode='markers',
                marker=dict(size=8, color=batted_balls['ExitSpeed'], colorscale='Viridis', showscale=True, colorbar=dict(title="Exit Velo (mph)")),
                hovertemplate="EV: %{marker.color:.1f} mph<br>PlateX: %{x:.2f}<br>PlateZ: %{z:.2f}<extra></extra>"
            ))
            fig_cp.update_layout(
                scene=dict(
                    xaxis_title='Plate Location Side (ft)',
                    yaxis_title='Contact Depth (ft)',
                    zaxis_title='Plate Location Height (ft)'
                ),
                height=500
            )
            st.plotly_chart(fig_cp, use_container_width=True)
        else:
            st.info("Contact depth coordinates not fully available in this dataset.")

    with tab2:
        st.markdown("### 2D Strike Zone Heatmaps")
        batted_balls['SZ_Zone'] = batted_balls.apply(lambda row: assign_sz_zone(row['PlateLocSide'], row['PlateLocHeight']), axis=1)
        
        col1, col2 = st.columns(2)
        
        def plot_2d_zone(df, metric, title, cmap):
            fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
            zone_avgs = df.groupby('SZ_Zone')[metric].mean()
            
            # Strike Zone Map bounds
            sz_configs = {
                'Z1': {'x': -0.833, 'y': 2.833}, 'Z2': {'x': -0.277, 'y': 2.833}, 'Z3': {'x': 0.277, 'y': 2.833},
                'Z4': {'x': -0.833, 'y': 2.166}, 'Z5': {'x': -0.277, 'y': 2.166}, 'Z6': {'x': 0.277, 'y': 2.166},
                'Z7': {'x': -0.833, 'y': 1.500}, 'Z8': {'x': -0.277, 'y': 1.500}, 'Z9': {'x': 0.277, 'y': 1.500}
            }
            w, h = 0.556, 0.667
            
            norm = mcolors.Normalize(vmin=df[metric].min(), vmax=df[metric].max())
            colormap = plt.get_cmap(cmap)
            
            for zone, coords in sz_configs.items():
                val = zone_avgs.get(zone, np.nan)
                color = colormap(norm(val)) if not pd.isna(val) else '#f1f5f9'
                text_val = f"{val:.1f}" if not pd.isna(val) else "-"
                
                rect = Rectangle((coords['x'], coords['y']), w, h, facecolor=color, edgecolor='black', lw=1.5)
                ax.add_patch(rect)
                ax.text(coords['x'] + w/2, coords['y'] + h/2, text_val, ha='center', va='center', 
                        fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(0.5, 4.5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontweight='bold', fontsize=14)
            return fig

        with col1:
            st.pyplot(plot_2d_zone(batted_balls, 'ExitSpeed', 'Average Exit Velocity by Zone', 'coolwarm'))
        with col2:
            st.pyplot(plot_2d_zone(batted_balls, 'Angle', 'Average Launch Angle by Zone', 'viridis'))

    with tab3:
        st.markdown("### Automated AI Player Profile & Scouting Report")
        if gemini_key:
            if st.button("Generate AI Scouting Report"):
                with st.spinner("Analyzing player data..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # Prepare data summary for the LLM
                        zone_evs = batted_balls.groupby('SZ_Zone')['ExitSpeed'].mean().to_dict()
                        zone_las = batted_balls.groupby('SZ_Zone')['Angle'].mean().to_dict()
                        
                        prompt = f"""
                        Act as a professional baseball hitting coordinator. Analyze the following Trackman batted ball data for {selected_batter} ({batter_side}). 
                        
                        Overall Session Metrics:
                        - Total Balls in Play: {total_bip}
                        - Average Exit Velocity: {avg_ev:.1f} mph
                        - Max Exit Velocity: {max_ev:.1f} mph
                        - Hard Hit Percentage (>90% Max EV): {hh_pct:.1f}%
                        - Average Launch Angle: {avg_la:.1f} degrees
                        - Average Distance: {avg_dist:.0f} feet
                        
                        Zone EV Averages (Z1-Z3 is Top, Z7-Z9 is Bottom):
                        {zone_evs}
                        
                        Zone Launch Angle Averages:
                        {zone_las}
                        
                        Provide a concise, 3-paragraph scouting report. 
                        Paragraph 1: Overall assessment of bat speed, power potential, and batted ball profile.
                        Paragraph 2: Identify specific strengths (e.g., pitches/zones they crush).
                        Paragraph 3: Identify specific deficiencies or mechanical flags based on zone performance and provide a targeted developmental focus.
                        """
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")
        else:
            st.warning("Please enter your Gemini API Key in the sidebar to unlock AI Scouting Analysis.")

    with tab4:
        st.markdown("### Raw Data View")
        st.dataframe(batted_balls[['PitchNo', 'Pitcher', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'HitSpinRate', 'PlateLocHeight', 'PlateLocSide']].sort_values(by='ExitSpeed', ascending=False), use_container_width=True)

        st.markdown("### Export Professional Report")
        def generate_pdf_report(df, batter_name, side, date):
            pdf_buffer = BytesIO()
            fig = plt.figure(figsize=(11, 8.5)) 
            fig.text(0.05, 0.92, f"{batter_name} | {side}", fontsize=24, fontweight='bold')
            fig.text(0.05, 0.88, f"OMAHA PLAYER DEVELOPMENT • Session: {date}", fontsize=12, color='gray')
            
            metrics = [
                ("BIP", f"{len(df)}"), ("MAX EV", f"{df['ExitSpeed'].max():.1f}"),
                ("AVG EV", f"{df['ExitSpeed'].mean():.1f}"), ("AVG LA", f"{df['Angle'].mean():.1f}"),
                ("AVG DIST", f"{df['Distance'].mean():.0f}")
            ]
            x_off = 0.05
            for label, val in metrics:
                fig.text(x_off, 0.80, label, fontsize=8, color='gray', fontweight='bold')
                fig.text(x_off, 0.76, val, fontsize=14, fontweight='bold')
                x_off += 0.15

            # Simplified spray chart for PDF
            ax_spray = fig.add_axes([0.05, 0.1, 0.4, 0.5])
            angles_rad = np.radians(np.linspace(-45, 45, 100))
            ax_spray.plot(380 * np.sin(angles_rad), 380 * np.cos(angles_rad), color='black')
            ax_spray.plot([0, 380*np.sin(np.radians(45))], [0, 380*np.cos(np.radians(45))], color='black')
            ax_spray.plot([0, 380*np.sin(np.radians(-45))], [0, 380*np.cos(np.radians(-45))], color='black')
            
            x = df['Distance'] * np.sin(np.radians(df['Direction']))
            y = df['Distance'] * np.cos(np.radians(df['Direction']))
            scatter = ax_spray.scatter(x, y, c=df['ExitSpeed'], cmap='coolwarm', vmin=70, vmax=105, s=30, edgecolor='black')
            ax_spray.set_xlim(-250, 250)
            ax_spray.set_ylim(-20, 400)
            ax_spray.set_aspect('equal')
            ax_spray.axis('off')
            ax_spray.set_title("Batted Ball Spray (Color = EV)")
            plt.colorbar(scatter, ax=ax_spray, fraction=0.046, pad=0.04)

            fig.savefig(pdf_buffer, format='pdf', dpi=150)
            plt.close(fig)
            return pdf_buffer.getvalue()

        pdf_bytes = generate_pdf_report(batted_balls, selected_batter, batter_side, selected_date)
        st.download_button(
            label="📄 Download PDF Session Report",
            data=pdf_bytes,
            file_name=f"{selected_batter.replace(' ', '_')}_Report.pdf",
            mime="application/pdf"
        )