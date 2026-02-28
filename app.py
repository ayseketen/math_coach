import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, datetime, timedelta
import os
import json
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

# -----------------------------
# Helpers
# -----------------------------
TOPICS_DEFAULT = [
    "Çarpanlar ve Katları",
    "EBOK-EKOK",
    "Üslü Sayılar",
    "Köklü Sayılar",
    "Veri Analizi (Grafik-Tablo)",
    "Olasılık",
    "Cebirsel İfadeler",
    "1.derece Denklemler ve Doğrusal İlişkiler"
]

QUOTES = [
    "Bugün yaptığın küçük bir adım, sınav gününde büyük bir fark yaratır.",
    "Disiplin, motivasyonun bittiği yerde devreye girer.",
    "Zorlandığın konu, puanı en hızlı yükselten konudur."
    "Hata yapmak öğrenmenin parçası — önemli olan geri dönmek.",
    "Bir soruyu çözemediysen, aslında yeni bir şey öğreniyorsun.",
    "Sen sıradan biri değilsin. Bu kadar yorulup hâlâ masaya oturabiliyorsan, içinde çoğu insanda olmayan bir güç var demektir."
    "Bugün ‘az’ bile olsa devam etmek, bırakmaktan kat kat iyidir.",
]


def go_main():
    st.session_state["topics"] = [
    t.strip() for t in st.session_state["topics_text"].splitlines() if t.strip()
]
    st.session_state["logged_in"] = True

def go_home():
    st.session_state["logged_in"] = False





def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_settings(d):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def daily_quote(today: date) -> str:
    idx = today.toordinal() % len(QUOTES)
    return QUOTES[idx]

def read_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=["date", "topic", "solved", "wrong"])

def append_history(rows: pd.DataFrame):
    df_old = read_history()
    df_new = pd.concat([df_old, rows], ignore_index=True)
    # aynı gün aynı konu varsa en son girilen kalsın (overwrite)
    df_new = (df_new
              .sort_values(["date"])
              .drop_duplicates(subset=["date", "topic"], keep="last"))
    df_new.to_csv(HISTORY_PATH, index=False)

def last_n_days(df: pd.DataFrame, n=7, end_date=None) -> pd.DataFrame:
    if df.empty:
        return df
    if end_date is None:
        end_date = date.today()
    start = end_date - timedelta(days=n-1)
    return df[(df["date"] >= start) & (df["date"] <= end_date)].copy()

def compute_topic_weights(history: pd.DataFrame, topics: list, lookback_days=7,
                          alpha=2.0, beta=1.0) -> pd.Series:
    """
    Yanlış oranı yüksek konulara daha fazla ağırlık verir.
    weight = 1 + alpha*(wrong/solved) + beta*(1-accuracy)
    """
    if history.empty:
        return pd.Series({t: 1.0 for t in topics})

    recent = last_n_days(history, n=lookback_days)
    if recent.empty:
        return pd.Series({t: 1.0 for t in topics})

    g = recent.groupby("topic", as_index=False).agg(
        solved=("solved", "sum"),
        wrong=("wrong", "sum"),
    )
    g["solved"] = g["solved"].clip(lower=0)
    g["wrong"] = g["wrong"].clip(lower=0)

    weights = {}
    for t in topics:
        row = g[g["topic"] == t]
        if row.empty or row.iloc[0]["solved"] == 0:
            weights[t] = 1.0
            continue
        solved = float(row.iloc[0]["solved"])
        wrong = float(row.iloc[0]["wrong"])
        accuracy = max(0.0, min(1.0, (solved - wrong) / solved))
        wrong_rate = wrong / solved
        weights[t] = 1.0 + alpha * wrong_rate + beta * (1.0 - accuracy)

    return pd.Series(weights)

def allocate_integers(total: int, weights: pd.Series) -> dict:
    """Ağırlıklara göre integer dağıtım (largest remainder)."""
    if total <= 0:
        return {k: 0 for k in weights.index}

    w = weights.clip(lower=0.0001)
    w = w / w.sum()
    raw = w * total
    floor = raw.astype(int)
    remainder = raw - floor
    allocated = floor.to_dict()

    left = total - sum(allocated.values())
    if left > 0:
        # en büyük kalanlara +1
        order = remainder.sort_values(ascending=False).index.tolist()
        for i in range(left):
            allocated[order[i % len(order)]] += 1
    return allocated

def read_targets(file) -> pd.DataFrame:
    """CSV/Excel hedef dosyasını okur; date ya da weekday ile çalışır."""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = [c.strip().lower() for c in df.columns]
    if "target" not in df.columns:
        raise ValueError("Dosyada 'target' sütunu olmalı.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[["date", "target"]].copy()
    elif "weekday" in df.columns:
        # Mon/Tue/... ya da Türkçe kabul etmek istersen burada genişletebilirsin
        df["weekday"] = df["weekday"].astype(str).str.strip()
        df = df[["weekday", "target"]].copy()
    else:
        raise ValueError("Dosyada 'date' veya 'weekday' sütunu olmalı.")

    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    return df

def todays_target(targets_df: pd.DataFrame, today: date) -> int:
    if targets_df is None or targets_df.empty:
        return 0
    cols = set(targets_df.columns)
    if "date" in cols:
        row = targets_df[targets_df["date"] == today]
        return int(row.iloc[0]["target"]) if not row.empty else 0
    # weekday
    wd = today.strftime("%a")  # Mon Tue...
    row = targets_df[targets_df["weekday"].str.lower() == wd.lower()]
    return int(row.iloc[0]["target"]) if not row.empty else 0

def analysis_day_quota(today: date, total_target: int, analysis_weekdays=("Fri", "Sun")) -> int:
    """Belirli günlerde yanlış analizi için hedefin bir kısmını ayır."""
    if total_target <= 0:
        return 0
    if today.strftime("%a") in analysis_weekdays:
        return max(10, int(total_target * 0.15))  # min 10 soru / %15
    return 0

def two_week_cycle_topic_of_day(day_index: int, topics: list) -> str:
    """14 günde 7 konuyu 2 kez döndür (odak konusu)."""
    return topics[day_index % len(topics)]

# -----------------------------
# UI bits: flying balloons
# -----------------------------
BALLOON_CSS = """
<style>
.balloon-wrap { position: relative; height: 240px; overflow:hidden; border-radius: 18px; }
.balloon { position:absolute; bottom:-80px; width:38px; height:48px; border-radius: 50% 50% 45% 45%;
          opacity:0.9; animation: floatUp 10s linear infinite; }
.balloon:after { content:""; position:absolute; left:50%; top:48px; width:2px; height:55px; background:rgba(0,0,0,0.15); transform:translateX(-50%); }
@keyframes floatUp { 0% { transform: translateY(0) translateX(0); } 100% { transform: translateY(-360px) translateX(40px);} }
.b1{ left:5%;  background: #ff6b6b; animation-duration: 9s; }
.b2{ left:18%; background: #feca57; animation-duration: 11s; animation-delay: -2s;}
.b3{ left:32%; background: #48dbfb; animation-duration: 10s; animation-delay: -4s;}
.b4{ left:48%; background: #1dd1a1; animation-duration: 12s; animation-delay: -1s;}
.b5{ left:65%; background: #5f27cd; animation-duration: 9.5s; animation-delay: -3s;}
.b6{ left:80%; background: #ff9ff3; animation-duration: 11.5s; animation-delay: -5s;}
.b7{ left:92%; background: #54a0ff; animation-duration: 10.5s; animation-delay: -6s;}
</style>
"""


def balloons_panel():
    st.markdown(BALLOON_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="balloon-wrap">
          <div class="balloon b1"></div>
          <div class="balloon b2"></div>
          <div class="balloon b3"></div>
          <div class="balloon b4"></div>
          <div class="balloon b5"></div>
          <div class="balloon b6"></div>
          <div class="balloon b7"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="460’a 💯 Gün", page_icon="🎈", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stDataEditor"] [class*="ag-theme"]{
        --ag-header-background-color: #2E86C1;
        --ag-header-foreground-color: #ffffff;
    }
    [data-testid="stDataEditor"] [class*="ag-header"]{
        color:#fff !important;
        font-weight:700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Session State init (GARANTİ) ----
if "topics" not in st.session_state:
    st.session_state["topics"] = TOPICS_DEFAULT.copy()

if "topics_text" not in st.session_state:
    st.session_state["topics_text"] = "\n".join(st.session_state["topics"])

settings = load_settings()
if "name" not in st.session_state:
    st.session_state["name"] = settings.get("name", "Nihalimm..")
if "exam_date" not in st.session_state:
    # Sen istersen sidebar'dan seçersin
    st.session_state["exam_date"] = datetime.strptime(settings.get("exam_date", "2026-06-14"), "%Y-%m-%d").date()
if "topics" not in st.session_state:
    st.session_state["topics"] = settings.get("topics", TOPICS_DEFAULT)
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "targets_df" not in st.session_state:
    st.session_state["targets_df"] = None

today = date.today()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.session_state["name"] = st.text_input("İsim", value=st.session_state["name"])
    st.session_state["exam_date"] = st.date_input("Sınav Tarihi", value=st.session_state["exam_date"])
    st.caption("Konu listesini istersen düzenleyebilirsin:")
    topics_text = st.text_area("Konular", value="\n".join(st.session_state["topics"]), height=160)
    st.session_state["topics"] = [t.strip() for t in topics_text.splitlines() if t.strip()][:8]

    st.divider()
    st.subheader("🎯 Haftalık Soru Hedefi")

# Haftalık hedefler session_state'te saklansın
    if "weekly_targets" not in st.session_state:
        st.session_state["weekly_targets"] = {
        "Mon": 55, "Tue": 40, "Wed": 45, "Thu": 35, "Fri": 60, "Sat": 40, "Sun": 50
        }

    weekdays = [("Mon", "Pazartesi"), ("Tue", "Salı"), ("Wed", "Çarşamba"), ("Thu", "Perşembe"),
                ("Fri", "Cuma"), ("Sat", "Cumartesi"), ("Sun", "Pazar")]

    for code, tr in weekdays:
        st.session_state["weekly_targets"][code] = st.number_input(
            f"{tr}", min_value=0, step=5,
            value=int(st.session_state["weekly_targets"].get(code, 0)),
            key=f"target_{code}"
        )

    st.caption("Bu hedefler o güne ait toplam soru hedefidir. Uygulama bunu konulara dağıtır.")

    if st.button("💾 Ayarları Kaydet"):
        save_settings({
            "name": st.session_state["name"],
            "exam_date": st.session_state["exam_date"].isoformat(),
            "topics": st.session_state["topics"]
        })
        st.success("Kaydedildi ✅")

    st.button("🏠 Ana Sayfa", on_click=go_home)

    st.divider()
    st.subheader("🗑️ Veri Yönetimi")

    if st.button("TÜM verileri sıfırla", type="secondary"):
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            f.write("date,topic,solved,wrong\n")
        st.success("Tüm geçmiş veriler sıfırlandı ✅")
        st.rerun()

# Entrance screen
if not st.session_state["logged_in"]:
    balloons_panel()
    st.markdown(f"## 🎈 Merhaba **{st.session_state['name']}**")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### ✨ _“{daily_quote(today)}”_")

    st.markdown("<br><br><br>", unsafe_allow_html=True)


    if st.button("Başla 🚀", on_click=go_main):
        st.balloons()

    
    st.stop()

# Main dashboard
history = read_history()
topics = st.session_state["topics"]
targets_df = st.session_state["targets_df"]

st.warning(f"##### ✨ _“{daily_quote(today)}”_")
st.markdown("<br>", unsafe_allow_html=True)
st.title("📚 460’a 💯 Gün")


# Countdown & quote
days_left = (st.session_state["exam_date"] - today).days
c1, c2 = st.columns(2)
c1.metric("⏳ Sınava kalan gün", days_left)
c2.metric("🗓️ Bugünün tarihi", today.strftime("%d.%m.%Y"))

# Targets & plan
wd = today.strftime("%a")  # Mon/Tue/...
total_target = int(st.session_state.get("weekly_targets", {}).get(wd, 0))
analysis_quota = analysis_day_quota(today, total_target)
topic_target_pool = max(0, total_target - analysis_quota)

weights = compute_topic_weights(history, topics, lookback_days=7, alpha=2.0, beta=1.0)
allocation = allocate_integers(topic_target_pool, weights)

st.subheader("🎯 Bugünkü Plan")
p1, p2 = st.columns([1.2, 1])

with p1:
    st.write(f"**Toplam hedef:** {total_target} soru")
    st.write(f"🧠 **Yanlış Analizi hedefi (hedefin içinden):** {analysis_quota}")


# Bugünkü hedefleri allocation'dan alıp tabloya koy
plan_df = pd.DataFrame({
    "Konu": topics,
    "Hedef": [int(allocation.get(t, 0)) for t in topics],
    "Çözülen": [0]*len(topics),
    "Yanlış": [0]*len(topics),
})

# Eğer bugün daha önce kayıt varsa, çözülen/yanlış alanlarını dolduralım
if not history.empty:
    todays = history[history["date"] == today].copy()
    if not todays.empty:
        m = todays.set_index("topic")[["solved", "wrong"]].to_dict(orient="index")
        plan_df["Çözülen"] = plan_df["Konu"].map(lambda t: int(m.get(t, {}).get("solved", 0)))
        plan_df["Yanlış"]  = plan_df["Konu"].map(lambda t: int(m.get(t, {}).get("wrong", 0)))

# Data editor: Hedef görünür ama değiştirilemez; çözülen/yanlış girilebilir

gb = GridOptionsBuilder.from_dataframe(plan_df)

gb.configure_column("Konu", editable=False)
gb.configure_column("Hedef", editable=False)
gb.configure_column("Çözülen", editable=True, type=["numericColumn"])
gb.configure_column("Yanlış", editable=True, type=["numericColumn"])

# ✅ Buraya ekle
gb.configure_grid_options(domLayout="autoHeight")

gridOptions = gb.build()

custom_css = {
    ".ag-header": {"background-color": "#2E86C1 !important"},
    ".ag-header-cell-label": {"color": "white !important", "font-weight": "700"},
}

grid_response = AgGrid(
    plan_df,
    gridOptions=gridOptions,
    update_mode=GridUpdateMode.VALUE_CHANGED,
    theme="streamlit",
    custom_css=custom_css,
    fit_columns_on_grid_load=True,
    # ✅ height'i KALDIR
)

edited = grid_response["data"]

# Yanlış analizi hedefin varsa ayrı küçük checkbox kalsın (istersen bunu da tabloya ekleriz)
st.checkbox("Yanlış analizi yaptım ✅", key=f"analysis_done_{today}")

# Kaydet
if st.button("Kaydet 💾", key=f"save_{today}"):
    rows = edited.copy()
    rows.rename(columns={"Çözülen": "solved", "Yanlış": "wrong", "Konu": "topic"}, inplace=True)
    rows["date"] = today
    rows = rows[["date", "topic", "solved", "wrong"]]

    # Güvenlik: yanlış çözüleni geçmesin
    rows["solved"] = pd.to_numeric(rows["solved"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    rows["wrong"]  = pd.to_numeric(rows["wrong"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    rows.loc[rows["wrong"] > rows["solved"], "wrong"] = rows["solved"]

    append_history(rows)
    st.success("Bugünkü giriş kaydedildi ✅")
    st.rerun()


    if analysis_quota > 0:
        st.markdown("#### 🧠 Yanlış Analizi")
        st.caption("Bugün yanlış analizi yaptın mı?")
        st.checkbox("Yanlış analizi yaptım ✅", key=f"analysis_done_{today}")

    if st.button("Kaydet 💾"):
        rows = pd.DataFrame([{
            "date": today,
            "topic": t,
            "solved": int(s),
            "wrong": int(w)
        } for (t, s, w) in inputs])
        append_history(rows)
        st.success("Bugünkü giriş kaydedildi ✅")
        st.rerun()

st.divider()

# -----------------------------
# Analytics
# -----------------------------
st.subheader("📊 İstatistikler")

if history.empty:
    st.info("Henüz veri yok. Bugün çözdüklerini girince grafikler burada oluşacak.")
else:
    # daily totals
    daily = history.groupby("date", as_index=False).agg(
        solved=("solved", "sum"),
        wrong=("wrong", "sum"),
    )
    daily["accuracy"] = (daily["solved"] - daily["wrong"]).clip(lower=0) / daily["solved"].replace(0, pd.NA)
    daily["accuracy"] = daily["accuracy"].fillna(0) * 100

    # last 14 days chart
    last14 = daily[daily["date"] >= (today - timedelta(days=13))].copy()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Günlük çözülen soru (son 14 gün)")
        chart1 = alt.Chart(last14).mark_line(point=True).encode(
            x=alt.X("date:T", title="Gün"),
            y=alt.Y("solved:Q", title="Çözülen Soru")
        ).properties(height=280)
        st.altair_chart(chart1, use_container_width=True)

    with colB:
        st.markdown("### Günlük doğruluk % (son 14 gün)")
        chart2 = alt.Chart(last14).mark_line(point=True).encode(
            x=alt.X("date:T", title="Gün"),
            y=alt.Y("accuracy:Q", title="Doğruluk (%)")
        ).properties(height=280)
        st.altair_chart(chart2, use_container_width=True)

    st.markdown("### Konu bazlı performans (son 7 gün)")
    recent = last_n_days(history, n=7, end_date=today)
    topic_perf = recent.groupby("topic", as_index=False).agg(
        solved=("solved", "sum"),
        wrong=("wrong", "sum"),
    )
    topic_perf["accuracy"] = ((topic_perf["solved"] - topic_perf["wrong"]).clip(lower=0)
                             / topic_perf["solved"].replace(0, pd.NA)).fillna(0) * 100

    bar = alt.Chart(topic_perf).mark_bar().encode(
        x=alt.X("topic:N", sort="-y", title="Konu"),
        y=alt.Y("accuracy:Q", title="Doğruluk (%)"),
        tooltip=["topic", "solved", "wrong", alt.Tooltip("accuracy:Q", format=".1f")]
    ).properties(height=320)
    st.altair_chart(bar, use_container_width=True)

    st.divider()
    st.markdown("### 📥 Verini dışa aktar")
    export_df = history.copy()
    export_df["date"] = export_df["date"].astype(str)
    st.download_button(
        "History CSV indir",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="math_history.csv",
        mime="text/csv"
    )