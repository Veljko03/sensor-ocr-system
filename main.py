from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import datetime
import sqlite3  # ← DODATO

app = FastAPI(title="Merna Mesta API")

# ═══════════════════════════════
# 1. INICIJALIZACIJA BAZE
# ═══════════════════════════════
def init_db():
    conn = sqlite3.connect("merenja.db")  # Kreira fajl merenja.db u istom folderu
    conn.execute("""
        CREATE TABLE IF NOT EXISTS merenja (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sat TEXT,
            vrednost REAL,
            vreme TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()  # Pokreće se jednom kada startuješ server

# ═══════════════════════════════
# 2. MODELI I MEMORIJA (ostaje isto)
# ═══════════════════════════════
class PodatakSaSata(BaseModel):
    sat: str
    vrednost: float

istorija_podataka: Dict[str, List[Dict]] = {"sat_1": [], "sat_2": []}
trenutne_vrednosti = {"sat_1": 0.0, "sat_2": 0.0}

# ═══════════════════════════════
# 3. HTML (ostaje isto kao pre)
# ═══════════════════════════════
HTML_STRANICA = """
<!DOCTYPE html>
<html>
<head>
    <title>Merna Mesta - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center; background-color: #eef2f3; margin: 0; padding: 20px; }
        .kontejner { display: flex; justify-content: center; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
        .kartica { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); width: 250px; }
        .grafikon-kartica { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); width: 80%; margin: 30px auto; }
        h1 { color: #2c3e50; }
        .vrednost { font-size: 32px; font-weight: bold;  margin-top: 10px; }
        button { margin-top: 20px; padding: 10px 25px; background: #e74c3c; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Merna Mesta - Kontrolna Tabla</h1>
    
    <div class="kontejner">
        <div class="kartica">
            <h3>Merno Mesto 1</h3>
            <div id="val-sat_1" class="vrednost" style="color: #2ecc71;">0.00</div>
        </div>
        <div class="kartica">
            <h3>Merno Mesto 2</h3>
            <div id="val-sat_2" class="vrednost" style="color: #3498db;">0.00</div>
        </div>
    </div>

    <div class="grafikon-kartica">
        <canvas id="mojGrafikon" style="width:100%; max-height:400px;"></canvas>
        <button onclick="ocistiBazu()">🗑️ Obriši sve podatke</button>
    </div>

    <script>
        const ctx = document.getElementById('mojGrafikon').getContext('2d');
        const grafikon = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Merno Mesto 1', data: [], borderColor: '#2ecc71', tension: 0.2, fill: false },
                    { label: 'Merno Mesto 2', data: [], borderColor: '#3498db', tension: 0.2, fill: false }
                ]
            },
            options: { scales: { x: { title: { display: true, text: 'Vreme' } }, y: { title: { display: true, text: 'Vrednost' } } } }
        });

        async function osveziPodatke() {
            try {
                const odgovor = await fetch('/api/podaci');
                const podaci = await odgovor.json();
                
                document.getElementById('val-sat_1').innerText = podaci.trenutno.sat_1.toFixed(2);
                document.getElementById('val-sat_2').innerText = podaci.trenutno.sat_2.toFixed(2);
                
                grafikon.data.labels = [];
                grafikon.data.datasets[0].data = [];
                grafikon.data.datasets[1].data = [];
                
                podaci.istorija.sat_1.forEach(p => {
                    grafikon.data.labels.push(p.vreme);
                    grafikon.data.datasets[0].data.push(p.vrednost);
                });
                
                podaci.istorija.sat_2.forEach((p, index) => {
                    if(grafikon.data.labels.length <= index) grafikon.data.labels.push(p.vreme);
                    grafikon.data.datasets[1].data.push(p.vrednost);
                });
                
                grafikon.update();
            } catch (greska) {
                console.error("Greška:", greska);
            }
        }

        async function ocistiBazu() {
            if (!confirm("Sigurno želiš da obrišeš sve podatke?")) return;
            await fetch('/api/obrisi', { method: 'DELETE' });
            alert("Baza obrisana!");
            osveziPodatke();
        }

        setInterval(osveziPodatke, 3000);
        osveziPodatke();
    </script>
</body>
</html>
"""

# ═══════════════════════════════
# 4. ENDPOINTOVI
# ═══════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def procitaj_dashboard():
    return HTML_STRANICA


@app.get("/api/podaci")
async def uzmi_podatke():
    # Čita iz baze umesto iz memorije
    conn = sqlite3.connect("merenja.db")
    rows = conn.execute(
        "SELECT sat, vrednost, vreme FROM merenja ORDER BY id DESC LIMIT 30"
    ).fetchall()
    conn.close()

    istorija = {"sat_1": [], "sat_2": []}
    trenutno = {"sat_1": 0.0, "sat_2": 0.0}

    for sat, vrednost, vreme in reversed(rows):
        if sat in istorija:
            istorija[sat].append({"vreme": vreme, "vrednost": vrednost})
            trenutno[sat] = vrednost  # poslednja vrednost = trenutna

    return {"trenutno": trenutno, "istorija": istorija}


@app.post("/update")
async def azuriraj_podatke(podatak: PodatakSaSata):
    if podatak.sat not in ["sat_1", "sat_2"]:
        raise HTTPException(status_code=400, detail="Koristi 'sat_1' ili 'sat_2'.")
    
    vreme_sada = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Čuva u bazu
    conn = sqlite3.connect("merenja.db")
    conn.execute(
        "INSERT INTO merenja (sat, vrednost, vreme) VALUES (?, ?, ?)",
        (podatak.sat, podatak.vrednost, vreme_sada)
    )
    conn.commit()
    conn.close()

    return {"status": "uspeh"}


# ═══════════════════════════════
# 5. NOVI ENDPOINT - BRISANJE BAZE
# ═══════════════════════════════
@app.delete("/api/obrisi")
async def obrisi_sve():
    conn = sqlite3.connect("merenja.db")
    conn.execute("DELETE FROM merenja")
    conn.commit()
    conn.close()
    return {"status": "baza obrisana"}