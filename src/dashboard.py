"""
Genera resultados/dashboard.html a partir de las tablas que deja
aprendizaje_supervisado.py. Sin dependencias extra: solo pandas + json.

Uso:  .venv/bin/python src/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / 'resultados'

# Paleta categorica validada para daltonismo (dataviz: pares adyacentes,
# claro y oscuro). Una familia = un color fijo, nunca reasignado.
FAMILIAS = ['Regresion', 'Arboles', 'Vectores', 'Red neuronal']


def cargar():
    covid = pd.read_csv(RESULTADOS / 'resultados_covid.csv')
    customer = pd.read_csv(RESULTADOS / 'resultados_customer.csv')
    enfoques = pd.read_csv(RESULTADOS / 'enfoques_customer.csv')
    resumen = json.loads((RESULTADOS / 'resumen.json').read_text(encoding='utf-8'))
    funciones = json.loads((RESULTADOS / 'funciones.json').read_text(encoding='utf-8'))
    limpieza = json.loads((RESULTADOS / 'limpieza.json').read_text(encoding='utf-8'))
    return {
        'covid': covid.to_dict(orient='records'),
        'customer': customer.to_dict(orient='records'),
        'enfoques': enfoques.to_dict(orient='records'),
        'resumen': resumen,
        'funciones': funciones,
        'limpieza': limpieza,
        'familias': FAMILIAS,
    }


PLANTILLA = """<meta charset="utf-8">
<title>Ocho modelos, dos datasets</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  color-scheme: light;
  --ground: #fcfcfb;
  --panel: #ffffff;
  --line: #e6e5e0;
  --line-strong: #cac9c1;
  --ink: #131311;
  --ink-2: #52514e;
  --ink-3: #72716a;
  --accent: #2a78d6;
  --critico: #c3403f;
  --serie-1: #2a78d6;
  --serie-2: #eb6834;
  --serie-3: #1baf7a;
  --serie-4: #eda100;
  --track: #edece7;
  --radio: 12px;
  --curva: cubic-bezier(0.22, 0.61, 0.36, 1);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #1a1a19;
    --panel: #232321;
    --line: #35352f;
    --line-strong: #4b4b43;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --ink-3: #94938a;
    --accent: #3987e5;
    --critico: #e66767;
    --serie-1: #3987e5;
    --serie-2: #d95926;
    --serie-3: #199e70;
    --serie-4: #c98500;
    --track: #2d2d2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #1a1a19;
  --panel: #232321;
  --line: #35352f;
  --line-strong: #4b4b43;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --ink-3: #94938a;
  --accent: #3987e5;
  --critico: #e66767;
  --serie-1: #3987e5;
  --serie-2: #d95926;
  --serie-3: #199e70;
  --serie-4: #c98500;
  --track: #2d2d2a;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: Newsreader, Georgia, "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
::selection { background: color-mix(in srgb, var(--accent) 22%, transparent); }

.hoja {
  max-width: 1080px;
  margin: 0 auto;
  padding-inline: 24px;
  padding-block: 56px 96px;
  display: flex;
  flex-direction: column;
  gap: 72px;
}

h1, h2, h3, .ui { font-family: Archivo, "Helvetica Neue", sans-serif; }
h1 {
  font-size: clamp(2.4rem, 6vw, 4rem);
  line-height: 1.02;
  letter-spacing: -0.035em;
  font-weight: 700;
  margin: 0;
  text-wrap: balance;
  max-width: 15ch;
}
h2 {
  font-size: clamp(1.35rem, 2.6vw, 1.75rem);
  letter-spacing: -0.025em;
  font-weight: 600;
  margin: 0;
  text-wrap: balance;
}
h3 {
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0;
}
p { margin: 0; max-width: 68ch; }
.prosa { color: var(--ink-2); }
.prosa strong { color: var(--ink); font-weight: 500; }

.dato {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}
.etiqueta {
  font-family: Archivo, sans-serif;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}

/* ---------- Cabecera y veredicto ---------- */
.cabecera { display: flex; flex-direction: column; gap: 20px; }
.entrada { color: var(--ink-2); font-size: 1.15rem; max-width: 62ch; }

.veredicto {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  border-top: 1px solid var(--line-strong);
  border-bottom: 1px solid var(--line);
  padding-block: 28px;
}
.veredicto > div { display: flex; flex-direction: column; gap: 10px; }
.veredicto > div + div {
  border-left: 1px solid var(--line);
  padding-left: 40px;
}
.ganador {
  font-family: Archivo, sans-serif;
  font-size: clamp(1.3rem, 3vw, 1.8rem);
  font-weight: 600;
  letter-spacing: -0.03em;
  line-height: 1.15;
}
.cifra { font-size: 0.95rem; color: var(--ink-2); }
.alerta { color: var(--critico); font-weight: 500; }

/* ---------- Secciones y graficos ---------- */
section { display: flex; flex-direction: column; gap: 24px; }
.encabezado-seccion { display: flex; flex-direction: column; gap: 8px; }

.grafico {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radio);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.barra-superior {
  display: flex;
  flex-wrap: wrap;
  gap: 16px 24px;
  align-items: baseline;
  justify-content: space-between;
}
.leyenda { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.leyenda span {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-family: Archivo, sans-serif;
  font-size: 0.78rem;
  color: var(--ink-2);
}
.punto { width: 10px; height: 10px; border-radius: 50%; flex: none; }

.selector { display: flex; gap: 4px; }
.selector button {
  font-family: Archivo, sans-serif;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--ink-2);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: var(--radio);
  padding: 5px 12px;
  cursor: pointer;
  transition: background 180ms var(--curva), color 180ms var(--curva), border-color 180ms var(--curva);
}
.selector button:hover { border-color: var(--line-strong); color: var(--ink); }
.selector button[aria-pressed="true"] {
  background: var(--ink);
  border-color: var(--ink);
  color: var(--ground);
}
:is(button, a, th[tabindex]):focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.filas { display: flex; flex-direction: column; gap: 2px; }
.fila {
  display: grid;
  grid-template-columns: 210px minmax(0, 1fr) 72px;
  align-items: center;
  gap: 16px;
  padding-block: 5px;
}
.fila:hover .pista { background: color-mix(in srgb, var(--ink) 7%, var(--track)); }
.nombre { display: flex; flex-direction: column; line-height: 1.25; }
.nombre b { font-family: Archivo, sans-serif; font-size: 0.86rem; font-weight: 500; }
.nombre small { font-size: 0.7rem; color: var(--ink-3); font-family: Archivo, sans-serif; }
.pista {
  position: relative;
  height: 22px;
  background: var(--track);
  border-radius: 0 4px 4px 0;
  transition: background 180ms var(--curva);
}
.barra {
  height: 100%;
  border-radius: 0 4px 4px 0;
  transition: width 620ms var(--curva);
}
.valor {
  font-family: "IBM Plex Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem;
  text-align: right;
  color: var(--ink);
}

.eje {
  display: grid;
  grid-template-columns: 210px minmax(0, 1fr) 72px;
  gap: 16px;
  align-items: start;
}
.marcas { position: relative; height: 18px; }
.marcas span {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  color: var(--ink-3);
  white-space: nowrap;
}
.marcas span:first-child { transform: none; }

.umbral {
  position: absolute;
  top: -6px;
  bottom: -6px;
  width: 0;
  border-left: 2px dashed var(--critico);
}
.nota-umbral {
  font-family: Archivo, sans-serif;
  font-size: 0.78rem;
  color: var(--critico);
  display: flex;
  align-items: center;
  gap: 8px;
}
.nota-umbral::before {
  content: "";
  width: 18px;
  border-top: 2px dashed var(--critico);
}
.pie-grafico { font-size: 0.85rem; color: var(--ink-3); max-width: 70ch; }

/* ---------- Tablas ---------- */
.contenedor-tabla { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radio); }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
caption {
  text-align: left;
  padding: 16px 20px 0;
  font-family: Archivo, sans-serif;
  font-size: 0.8rem;
  color: var(--ink-3);
}
th, td { padding: 10px 20px; text-align: left; white-space: nowrap; }
thead th {
  font-family: Archivo, sans-serif;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-3);
  font-weight: 600;
  border-bottom: 1px solid var(--line-strong);
}
tbody td { border-bottom: 1px solid var(--line); }
tbody tr:last-child td { border-bottom: 0; }
#tabla-covid tbody tr:first-child td,
#tabla-customer tbody tr:first-child td,
tbody tr.destacada td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.titulo-suelto { margin-top: 16px; }
td.si { color: var(--serie-3); font-weight: 500; }
td.no { color: var(--ink-3); }
td.num {
  font-family: "IBM Plex Mono", monospace;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
th.num { text-align: right; }
/* columnas largas que deben envolver en vez de empujar la tabla fuera de la vista */
#tabla-origen td:last-child { white-space: normal; overflow-wrap: anywhere; max-width: 230px; font-size: 0.74rem; }
#tabla-enfoques td:nth-child(2) { white-space: normal; min-width: 190px; }
.familia-celda { display: inline-flex; align-items: center; gap: 8px; }

/* ---------- Trazabilidad de la limpieza ---------- */
.limpieza { display: flex; flex-direction: column; gap: 32px; }
.rastro {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radio);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.rastro h3 { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; }
.rastro h3 span { font-family: "IBM Plex Mono", monospace; font-weight: 400; color: var(--ink-3); font-size: 0.8rem; }
.embudo { display: flex; flex-direction: column; gap: 2px; }
.tramo {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr) auto;
  align-items: center;
  gap: 16px;
  padding-block: 7px;
  border-bottom: 1px solid var(--line);
}
.tramo:last-child { border-bottom: 0; }
.tramo > b { font-family: Archivo, sans-serif; font-size: 0.84rem; font-weight: 500; }
.barra-tramo { height: 14px; background: var(--track); border-radius: 0 3px 3px 0; position: relative; }
.barra-tramo i { display: block; height: 100%; background: var(--serie-1); border-radius: 0 3px 3px 0; }
.barra-tramo u {
  position: absolute; top: 0; height: 100%;
  background: var(--critico); text-decoration: none; border-radius: 0 3px 3px 0;
}
.conteo {
  font-family: "IBM Plex Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem;
  color: var(--ink-2);
  white-space: nowrap;
}
.conteo b { color: var(--critico); font-weight: 500; }
.evidencia {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
.evidencia > div {
  border: 1px solid var(--line);
  border-radius: var(--radio);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.evidencia h4 {
  margin: 0;
  font-family: Archivo, sans-serif;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.campo {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.76rem;
  color: var(--ink-2);
}
.campo b { color: var(--ink); font-weight: 500; text-align: right; overflow-wrap: anywhere; }
.campo.encendido b { color: var(--serie-3); }
.flecha { font-family: "IBM Plex Mono", monospace; font-size: 0.76rem; color: var(--ink-3); }

/* ---------- Funciones ajustadas ---------- */
.funciones { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
.funcion {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radio);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.funcion h3 { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
.formula {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  color: var(--ink-2);
  background: color-mix(in srgb, var(--ink) 4%, transparent);
  border-radius: var(--radio);
  padding: 10px 14px;
  line-height: 1.7;
  overflow-wrap: break-word;
}
.funcion svg { width: 100%; height: auto; display: block; overflow: hidden; }
.funcion figcaption { font-size: 0.88rem; color: var(--ink-2); }
.cifras-linea {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 20px;
  font-family: Archivo, sans-serif;
  font-size: 0.78rem;
  color: var(--ink-2);
}
.cifras-linea b { font-family: "IBM Plex Mono", monospace; font-weight: 500; color: var(--ink); }
.eje-titulo { font-family: Archivo, sans-serif; font-size: 0.68rem; fill: var(--ink-3); }
.marca-svg { font-family: "IBM Plex Mono", monospace; font-size: 9px; fill: var(--ink-3); }

.diagnostico { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 48px; }
.diagnostico > div {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-block: 24px;
  border-top: 1px solid var(--line);
}
.diagnostico h3 { margin-bottom: 6px; }
.par {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
  font-family: Archivo, sans-serif;
  font-size: 0.82rem;
  color: var(--ink-2);
  border-bottom: 1px dotted var(--line-strong);
  padding-bottom: 4px;
}
.par .dato { font-size: 0.95rem; color: var(--ink); }
.diagnostico p { font-size: 0.9rem; color: var(--ink-2); margin-top: 8px; }
.diagnostico .alerta-suave { color: var(--critico); }

.metodo { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 32px 48px; }
.metodo > div { display: flex; flex-direction: column; gap: 8px; }
.metodo p { font-size: 0.92rem; color: var(--ink-2); }

footer {
  border-top: 1px solid var(--line);
  padding-top: 24px;
  font-size: 0.82rem;
  color: var(--ink-3);
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
}

@media (max-width: 760px) {
  .veredicto { grid-template-columns: 1fr; gap: 24px; }
  .veredicto > div + div { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 24px; }
  .metodo, .diagnostico { grid-template-columns: 1fr; }
  .fila, .eje { grid-template-columns: minmax(0, 1fr) 64px; }
  .nombre { grid-column: 1 / -1; margin-bottom: 2px; }
  .eje .marcas { grid-column: 1; }
  .eje > span:first-child { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 1ms !important; }
}
</style>

<div class="hoja">

  <header class="cabecera">
    <h1>Ocho modelos, dos datasets</h1>
    <p class="entrada">Dos variantes de cada familia vista en clase &mdash; regresi&oacute;n, &aacute;rboles,
    vectores de soporte y redes neuronales &mdash; entrenadas sobre los mismos datos limpios,
    con la misma partici&oacute;n 80/20 y la misma semilla. Estas son las cifras que decide cada gr&aacute;fico.</p>
  </header>

  <div class="veredicto">
    <div>
      <span class="etiqueta">covid_19_data.csv &middot; regresi&oacute;n</span>
      <span class="ganador" id="ganador-covid"></span>
      <span class="cifra dato" id="cifra-covid"></span>
    </div>
    <div>
      <span class="etiqueta">Customer.csv &middot; clasificaci&oacute;n</span>
      <span class="ganador" id="ganador-customer"></span>
      <span class="cifra alerta" id="cifra-customer"></span>
    </div>
  </div>

  <section>
    <div class="encabezado-seccion">
      <h2>Qu&eacute; tan bien predice cada modelo las muertes por covid</h2>
      <p class="prosa">Objetivo: <strong>Deaths</strong> en escala log1p, a partir de casos confirmados,
      recuperados, d&iacute;a y pa&iacute;s. R&sup2; mide qu&eacute; fracci&oacute;n de la variaci&oacute;n explica el modelo;
      RMSE y MAE miden el error en la misma escala del objetivo.</p>
    </div>
    <div class="grafico">
      <div class="barra-superior">
        <div class="leyenda" id="leyenda-covid"></div>
        <div class="selector" id="selector-covid">
          <button type="button" data-metrica="R2" aria-pressed="true">R&sup2;</button>
          <button type="button" data-metrica="RMSE" aria-pressed="false">RMSE</button>
          <button type="button" data-metrica="MAE" aria-pressed="false">MAE</button>
        </div>
      </div>
      <div class="filas" id="filas-covid"></div>
      <div class="eje"><span></span><div class="marcas" id="marcas-covid"></div><span></span></div>
      <p class="pie-grafico" id="pie-covid"></p>
    </div>
  </section>

  <section>
    <div class="encabezado-seccion">
      <h2>Por qu&eacute; ning&uacute;n modelo sirve para Customer.csv</h2>
      <p class="prosa">Objetivo: <strong>Segment</strong> a partir de edad, ciudad, estado, c&oacute;digo postal y
      regi&oacute;n. La l&iacute;nea roja es el baseline: la exactitud que se obtiene respondiendo siempre
      &laquo;Consumer&raquo;, sin entrenar nada. Un modelo a la izquierda de esa l&iacute;nea no aprendi&oacute; nada &uacute;til.</p>
    </div>
    <div class="grafico">
      <div class="barra-superior">
        <div class="leyenda" id="leyenda-customer"></div>
        <span class="nota-umbral" id="nota-umbral"></span>
      </div>
      <div class="filas" id="filas-customer"></div>
      <div class="eje"><span></span><div class="marcas" id="marcas-customer"></div><span></span></div>
      <p class="pie-grafico" id="pie-customer"></p>
    </div>
  </section>

  <section>
    <div class="encabezado-seccion">
      <h2>Lo que cuesta cada punto de precisi&oacute;n</h2>
      <p class="prosa">Segundos de entrenamiento sobre la muestra de covid, en la misma m&aacute;quina.
      El modelo ganador tarda menos que los dos que quedaron segundo y tercero.</p>
    </div>
    <div class="grafico">
      <div class="barra-superior">
        <div class="leyenda" id="leyenda-costo"></div>
      </div>
      <div class="filas" id="filas-costo"></div>
      <div class="eje"><span></span><div class="marcas" id="marcas-costo"></div><span></span></div>
      <p class="pie-grafico">Medido con <span class="dato">time.perf_counter()</span> alrededor de
      <span class="dato">fit()</span> y <span class="dato">predict()</span>, escalado incluido.</p>
    </div>
  </section>

  <section>
    <div class="encabezado-seccion">
      <h2>Las funciones que ajusta cada familia</h2>
      <p class="prosa">Las mismas rectas, curvas y fronteras de los apuntes, pero ajustadas sobre
      3&nbsp;000 registros reales de covid en vez de dibujadas de ejemplo. El eje horizontal es
      siempre <span class="dato">log(1+Confirmed)</span>; el vertical, lo que cada modelo predice.</p>
    </div>
    <div class="funciones">
      <figure class="funcion">
        <h3>Recta contra curva <span class="etiqueta">Apunte 1.6</span></h3>
        <div class="formula">recta &nbsp; h(x) = w&middot;x + b<br>polin&oacute;mica &nbsp; h(x) = w&#8321;x + w&#8322;x&sup2; + w&#8323;x&sup3; + b</div>
        <div id="svg-recta"></div>
        <div class="cifras-linea" id="cifras-recta"></div>
        <figcaption>La recta es el modelo lineal del apunte 1.6. La curva de grado 3 sigue siendo
        regresi&oacute;n lineal &mdash; los pesos entran de forma lineal &mdash;; lo que cambia es que suma
        potencias de x, y por eso puede doblarse donde la recta no llega.</figcaption>
      </figure>

      <figure class="funcion">
        <h3>Escalones del &aacute;rbol <span class="etiqueta">Apunte 1.7</span></h3>
        <div class="formula">si x &lt; corte&#8321; &rarr; valor&#8321;<br>si no, si x &lt; corte&#8322; &rarr; valor&#8322; &hellip;</div>
        <div id="svg-escalones"></div>
        <div class="cifras-linea" id="cifras-escalones"></div>
        <figcaption>El &aacute;rbol no ajusta ninguna funci&oacute;n continua: parte el eje en tramos y predice
        un valor constante dentro de cada uno. Esa es la raz&oacute;n de que gane en covid &mdash; puede
        seguir cualquier forma &mdash; y de que falle al extrapolar fuera del rango que vio.</figcaption>
      </figure>

      <figure class="funcion">
        <h3>La sigmoide <span class="etiqueta">Apunte 1.6</span></h3>
        <div class="formula">P(Y=1|x) = e^(wx+b) / (1 + e^(wx+b))</div>
        <div id="svg-sigmoide"></div>
        <div class="cifras-linea" id="cifras-sigmoide"></div>
        <figcaption>Clasificar si un registro tiene muchas muertes. La regresi&oacute;n log&iacute;stica no
        devuelve una clase sino una probabilidad entre 0 y 1; el umbral de 0.5 es lo que la
        convierte en decisi&oacute;n, y el punto donde la curva lo cruza es la frontera del modelo.</figcaption>
      </figure>

      <figure class="funcion">
        <h3>Hiperplano y m&aacute;rgenes <span class="etiqueta">Apunte 1.7</span></h3>
        <div class="formula">w&middot;x + b = 0 &nbsp; frontera<br>w&middot;x + b = &plusmn;1 &nbsp; m&aacute;rgenes</div>
        <div id="svg-hiperplano"></div>
        <div class="cifras-linea" id="cifras-hiperplano"></div>
        <figcaption>El SVM lineal busca la recta que separa las clases dejando el mayor margen
        posible. Solo los puntos sobre los m&aacute;rgenes &mdash; los vectores de soporte, marcados con
        anillo &mdash; definen d&oacute;nde queda la frontera; el resto de los datos no la mueve.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <div class="encabezado-seccion">
      <h2>&iquest;Y si entrenamos un modelo que s&iacute; se ajuste?</h2>
      <p class="prosa">Se puede, y ah&iacute; est&aacute; el punto: ajustarse a los datos de entrenamiento y
      predecir datos nuevos son dos cosas distintas. Cuatro experimentos, en
      <span class="dato">src/diagnostico_customer.py</span>.</p>
    </div>
    <div class="diagnostico">
      <div>
        <h3>Un &aacute;rbol sin l&iacute;mite se ajusta casi perfecto</h3>
        <div class="par"><span>Entrenamiento</span><span class="dato">0.9790</span></div>
        <div class="par"><span>Prueba</span><span class="dato">0.4323</span></div>
        <p>83 niveles y 310 hojas para 620 filas: memoriza lo que ya vio y falla en lo que no.
        Ni siquiera llega a 1.0000 porque hay clientes con la misma edad y ubicaci&oacute;n y distinto
        segmento.</p>
      </div>
      <div>
        <h3>Buscar hiperpar&aacute;metros no mueve el techo</h3>
        <div class="par"><span>Mejor de 24 combinaciones (CV)</span><span class="dato">0.5145</span></div>
        <div class="par"><span>Esa configuraci&oacute;n en prueba</span><span class="dato">0.5032</span></div>
        <p>La ganadora es <span class="dato">max_depth=3</span>: un bosque tan podado que responde
        casi siempre la clase mayoritaria. La b&uacute;squeda reproduce el baseline, no lo supera.</p>
      </div>
      <div>
        <h3>Con las etiquetas barajadas rinde igual</h3>
        <div class="par"><span>Etiquetas reales</span><span class="dato">0.4903</span></div>
        <div class="par"><span>Etiquetas al azar (30 corridas)</span><span class="dato">0.4983</span></div>
        <p class="alerta-suave">25 de 30 modelos entrenados con etiquetas al azar igualaron o
        superaron al modelo real. Romper la relaci&oacute;n no empeora nada porque no hab&iacute;a
        relaci&oacute;n que romper.</p>
      </div>
      <div>
        <h3>El mismo c&oacute;digo s&iacute; aprende Region</h3>
        <div class="par"><span>Region, &aacute;rbol sin l&iacute;mite</span><span class="dato">1.0000</span></div>
        <div class="par"><span>Segment, el mismo &aacute;rbol</span><span class="dato">0.4323</span></div>
        <p>El estado determina la regi&oacute;n &mdash; California es West, Kentucky es South &mdash; y el
        &aacute;rbol la aprende entera. El pipeline funciona: lo que falta es la variable, no el
        algoritmo.</p>
      </div>
    </div>
    <h3 class="titulo-suelto">&iquest;Y si limpi&aacute;ramos u organiz&aacute;ramos el dataset de otra forma?</h3>
    <p class="prosa">Ocho maneras distintas de preparar el mismo archivo, cada una con tres modelos
    entrenados encima. Se compara el mejor de los tres contra el baseline de ese enfoque.</p>
    <div class="contenedor-tabla">
      <table id="tabla-enfoques">
        <caption>Cada fila cambia la limpieza, la codificaci&oacute;n o la etiqueta. Solo la &uacute;ltima
        cambia la etiqueta.</caption>
      </table>
    </div>
    <p class="prosa">Siete formas distintas de organizar los datos para predecir
    <strong>Segment</strong>, cero que superen su baseline &mdash; y varias que lo empatan exacto
    porque el modelo termina respondiendo siempre la clase mayoritaria. El &uacute;nico enfoque que
    aprende algo es el que cambia la etiqueta a <strong>Region</strong>, donde el mismo c&oacute;digo
    llega a exactitud perfecta. Para predecir Segment har&iacute;an falta variables de comportamiento de
    compra &mdash; volumen, frecuencia, categor&iacute;as, tipo de env&iacute;o &mdash; que viven en la tabla
    de &oacute;rdenes del dataset original, no en este archivo.</p>
  </section>

  <section>
    <h2>Tablas completas</h2>
    <div class="contenedor-tabla">
      <table id="tabla-covid">
        <caption>covid_19_data.csv &mdash; regresi&oacute;n de Deaths (log1p). Mayor R&sup2; es mejor; menor RMSE y MAE es mejor.</caption>
      </table>
    </div>
    <div class="contenedor-tabla">
      <table id="tabla-customer">
        <caption>Customer.csv &mdash; clasificaci&oacute;n de Segment. Mayor exactitud y F1 macro es mejor.</caption>
      </table>
    </div>
  </section>

  <section>
    <div class="encabezado-seccion">
      <h2>De d&oacute;nde sale cada n&uacute;mero de esta p&aacute;gina</h2>
      <p class="prosa">Esta p&aacute;gina no lee los CSV al abrirse: es la foto de una corrida.
      Lo que sigue es el rastro de esa corrida &mdash; qu&eacute; archivos se leyeron, con qu&eacute; huella
      digital, y qu&eacute; elimin&oacute; cada proceso de limpieza. Para verlo ejecut&aacute;ndose,
      <span class="dato">src/demo.py</span> repite el recorrido paso a paso en la terminal.</p>
    </div>

    <div class="contenedor-tabla">
      <table id="tabla-origen">
        <caption>Archivos le&iacute;dos. El SHA-256 se puede comprobar con
        <span class="dato">sha256sum</span> sobre el archivo: si coincide, estos resultados
        salieron de ese archivo exacto.</caption>
      </table>
    </div>

    <div class="limpieza" id="limpieza"></div>
  </section>

  <section>
    <h2>C&oacute;mo se prepararon los datos</h2>
    <div class="metodo">
      <div>
        <h3>Columnas descartadas</h3>
        <p>Customer ID y Customer Name tienen un valor distinto por fila: convertidas a variables
        dummy generar&iacute;an 793 columnas sin informaci&oacute;n. Country es constante. En covid se elimin&oacute;
        la columna Province/State, no las filas: est&aacute; vac&iacute;a en 78&nbsp;103 registros y un
        <span class="dato">dropna()</span> habr&iacute;a borrado un cuarto del dataset.</p>
      </div>
      <div>
        <h3>Ruido y excepciones</h3>
        <p>El archivo completo de covid trae conteos negativos &mdash; el m&iacute;nimo de Confirmed
        es &minus;302&nbsp;844 &mdash; que el c&oacute;digo descarta por imposibles; en esta muestra
        no cay&oacute; ninguno. Los conteos se transforman con
        log1p antes del filtro de rango intercuart&iacute;lico: sin eso, la cola larga hace que el
        IQR marque como at&iacute;pico a casi todo el dataset.</p>
      </div>
      <div>
        <h3>Escalado y partici&oacute;n</h3>
        <p>Cada modelo corre dentro de un pipeline con StandardScaler, obligatorio para SVM,
        redes neuronales y modelos lineales regularizados. Partici&oacute;n 80/20 con
        <span class="dato">random_state=42</span>, estratificada por clase en Customer.csv.</p>
      </div>
      <div>
        <h3>Tama&ntilde;o de la muestra</h3>
        <p>De los 306&nbsp;429 registros de covid se muestrean
        <span class="dato" id="muestra-covid"></span> filas: SVM y redes neuronales escalan de forma
        cuadr&aacute;tica y el dataset completo no termina en un tiempo razonable. La constante
        <span class="dato">COVID_MUESTRA</span> controla ese valor.</p>
      </div>
    </div>
  </section>

  <footer>
    <span>Datos: Customer.csv y covid_19_data.csv (Johns Hopkins CSSE, corte 2021).</span>
    <span>scikit-learn 1.9.1 &middot; semilla 42</span>
    <span id="pie-corrida"></span>
  </footer>
</div>

<script>
const DATOS = %%DATOS%%;
const COLORES = ["var(--serie-1)", "var(--serie-2)", "var(--serie-3)", "var(--serie-4)"];
const color = (familia) => COLORES[DATOS.familias.indexOf(familia)] || "var(--serie-1)";
const fmt = (v, d = 4) => v.toFixed(d);

function leyenda(destino) {
  document.getElementById(destino).innerHTML = DATOS.familias
    .map((f) => `<span><i class="punto" style="background:${color(f)}"></i>${f}</span>`)
    .join("");
}

function marcas(destino, max, decimales) {
  const pasos = [0, 0.25, 0.5, 0.75, 1];
  document.getElementById(destino).innerHTML = pasos
    .map((p) => `<span style="left:${p * 100}%">${(max * p).toFixed(decimales)}</span>`)
    .join("");
}

function filas(destino, registros, metrica, max, umbral) {
  const html = registros.map((r) => {
    const ancho = Math.max(0, (r[metrica] / max) * 100);
    const marca = umbral
      ? `<span class="umbral" style="left:${(umbral / max) * 100}%"></span>`
      : "";
    return `<div class="fila">
      <span class="nombre"><b>${r.Modelo}</b><small>${r.Familia}</small></span>
      <div class="pista">${marca}<div class="barra" style="width:${ancho}%;background:${color(r.Familia)}"></div></div>
      <span class="valor">${fmt(r[metrica], metrica === "Segundos" ? 2 : 4)}</span>
    </div>`;
  });
  document.getElementById(destino).innerHTML = html.join("");
}

function tabla(id, registros, columnas, destacada = null) {
  const tabla = document.getElementById(id);
  const encabezado = `<thead><tr>${columnas
    .map((c) => `<th class="${c.num ? "num" : ""}">${c.titulo}</th>`)
    .join("")}</tr></thead>`;
  const cuerpo = registros
    .map((r) => `<tr class="${destacada && destacada(r) ? "destacada" : ""}">${columnas
      .map((c) => {
        if (c.clave === "Familia") {
          return `<td><span class="familia-celda"><i class="punto" style="background:${color(r.Familia)}"></i>${r.Familia}</span></td>`;
        }
        const v = r[c.clave];
        if (c.estado) return `<td class="${v === "si" ? "si" : "no"}">${v === "si" ? "s\u00ed" : "no"}</td>`;
        return c.num
          ? `<td class="num">${fmt(v, c.decimales ?? 4)}</td>`
          : `<td>${v}</td>`;
      })
      .join("")}</tr>`)
    .join("");
  tabla.insertAdjacentHTML("beforeend", encabezado + `<tbody>${cuerpo}</tbody>`);
}

// --- Motor de graficos XY (SVG) ------------------------------------------
// Una sola escala por eje, calculada sobre todas las series que se dibujan,
// para que ninguna marca quede fuera del area.
const LIENZO = { ancho: 460, alto: 300, izq: 44, der: 14, arriba: 26, abajo: 34 };

function escalas(series, dominioY) {
  const xs = series.flatMap((s) => s.map((p) => p[0]));
  const ys = dominioY || series.flatMap((s) => s.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const ancho = LIENZO.ancho - LIENZO.izq - LIENZO.der;
  const alto = LIENZO.alto - LIENZO.arriba - LIENZO.abajo;
  return {
    x0, x1, y0, y1,
    ex: (v) => LIENZO.izq + ((v - x0) / (x1 - x0 || 1)) * ancho,
    ey: (v) => LIENZO.arriba + alto - ((v - y0) / (y1 - y0 || 1)) * alto,
  };
}

function ejes(e, tituloX, tituloY, decimales = 1, marcasYfijas = null) {
  const marcasX = [0, 0.25, 0.5, 0.75, 1].map((t) => e.x0 + t * (e.x1 - e.x0));
  const marcasY = marcasYfijas || [0, 0.25, 0.5, 0.75, 1].map((t) => e.y0 + t * (e.y1 - e.y0));
  const base = LIENZO.alto - LIENZO.abajo;
  const rejilla = marcasY.map((v) =>
    `<line x1="${LIENZO.izq}" y1="${e.ey(v).toFixed(1)}" x2="${LIENZO.ancho - LIENZO.der}" y2="${e.ey(v).toFixed(1)}" stroke="var(--line)" stroke-width="1" fill="none"></line>`).join("");
  const etiquetasY = marcasY.map((v) =>
    `<text class="marca-svg" x="${LIENZO.izq - 6}" y="${(e.ey(v) + 3).toFixed(1)}" text-anchor="end">${v.toFixed(decimales)}</text>`).join("");
  const etiquetasX = marcasX.map((v, i) =>
    `<text class="marca-svg" x="${e.ex(v).toFixed(1)}" y="${base + 14}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${v.toFixed(decimales)}</text>`).join("");
  const titulos = `<text class="eje-titulo" x="${LIENZO.ancho - LIENZO.der}" y="${LIENZO.alto - 2}" text-anchor="end">${tituloX}</text>`
    + `<text class="eje-titulo" x="0" y="12" text-anchor="start">${tituloY}</text>`;
  return rejilla + etiquetasY + etiquetasX + titulos;
}

const trazo = (e, puntos, color, ancho = 2, guion = null) =>
  `<polyline points="${puntos.map((p) => `${e.ex(p[0]).toFixed(1)},${e.ey(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="${ancho}" stroke-linejoin="round" stroke-linecap="round"${guion ? ` stroke-dasharray="${guion}"` : ""}></polyline>`;

const nube = (e, puntos, color, radio = 2.4, opacidad = 0.5) =>
  puntos.map((p) => `<circle cx="${e.ex(p[0]).toFixed(1)}" cy="${e.ey(p[1]).toFixed(1)}" r="${radio}" fill="${color}" fill-opacity="${opacidad}"></circle>`).join("");

const anillos = (e, puntos, color) =>
  puntos.map((p) => `<circle cx="${e.ex(p[0]).toFixed(1)}" cy="${e.ey(p[1]).toFixed(1)}" r="4.5" fill="none" stroke="${color}" stroke-width="1.5"></circle>`).join("");

function lienzo(destino, fondo, marcas, titulo) {
  const id = `recorte-${destino}`;
  const rect = `<rect x="${LIENZO.izq}" y="${LIENZO.arriba}" width="${LIENZO.ancho - LIENZO.izq - LIENZO.der}" height="${LIENZO.alto - LIENZO.arriba - LIENZO.abajo}"></rect>`;
  document.getElementById(destino).innerHTML =
    `<svg viewBox="0 0 ${LIENZO.ancho} ${LIENZO.alto}" role="img" aria-label="${titulo}">`
    + `<defs><clipPath id="${id}">${rect}</clipPath></defs>`
    + fondo
    + `<g clip-path="url(#${id})">${marcas}</g></svg>`;
}

const cifras = (destino, pares) => {
  document.getElementById(destino).innerHTML = pares
    .map(([etiqueta, valor]) => `<span>${etiqueta} <b>${valor}</b></span>`).join("");
};

// --- Grafico 1: recta contra curva ---------------------------------------
const F = DATOS.funciones;
{
  const d = F.recta_curva;
  const e = escalas([d.puntos, d.recta, d.curva]);
  lienzo("svg-recta",
    ejes(e, "log(1+Confirmed)", "log(1+Deaths)"),
    nube(e, d.puntos, "var(--ink-3)", 2.2, 0.35)
    + trazo(e, d.recta, "var(--serie-1)", 2.5)
    + trazo(e, d.curva, "var(--serie-2)", 2.5),
    "Dispersion de casos confirmados contra muertes, con la recta de la regresion lineal y la curva polinomica de grado 3");
  cifras("cifras-recta", [
    [`<i class="punto" style="background:var(--serie-1);display:inline-block"></i> Recta R²`, d.r2_recta],
    [`<i class="punto" style="background:var(--serie-2);display:inline-block"></i> Curva R²`, d.r2_curva],
    ["Ecuación", `y = ${d.pendiente}x ${d.sesgo < 0 ? "-" : "+"} ${Math.abs(d.sesgo)}`],
  ]);
}

// --- Grafico 2: escalones del arbol --------------------------------------
{
  const d = F.escalones;
  const e = escalas([d.puntos, d.escalera]);
  lienzo("svg-escalones",
    ejes(e, "log(1+Confirmed)", "log(1+Deaths)"),
    nube(e, d.puntos, "var(--ink-3)", 2.2, 0.35)
    + trazo(e, d.escalera, "var(--serie-3)", 2.5),
    "La misma dispersion ajustada por un arbol de decision, que forma escalones en vez de una curva");
  cifras("cifras-escalones", [
    ["Escalones", d.hojas], ["R²", d.r2],
    ["Cortes en x", d.cortes.slice(0, 4).join(", ")],
  ]);
}

// --- Grafico 3: sigmoide -------------------------------------------------
{
  const d = F.sigmoide;
  const e = escalas([d.curva, d.positivos, d.negativos], [-0.05, 1.05]);
  const medio = `<line x1="${LIENZO.izq}" y1="${e.ey(0.5).toFixed(1)}" x2="${LIENZO.ancho - LIENZO.der}" y2="${e.ey(0.5).toFixed(1)}" stroke="var(--critico)" stroke-width="1.5" stroke-dasharray="5 4" fill="none"></line>`;
  const corte = `<line x1="${e.ex(d.umbral).toFixed(1)}" y1="${LIENZO.arriba}" x2="${e.ex(d.umbral).toFixed(1)}" y2="${LIENZO.alto - LIENZO.abajo}" stroke="var(--critico)" stroke-width="1.5" stroke-dasharray="5 4" fill="none"></line>`;
  lienzo("svg-sigmoide",
    ejes(e, "log(1+Confirmed)", "P(muchas muertes)", 2, [0, 0.25, 0.5, 0.75, 1]),
    nube(e, d.negativos, "var(--serie-4)", 2.6, 0.35)
    + nube(e, d.positivos, "var(--serie-1)", 2.6, 0.35)
    + medio + corte
    + trazo(e, d.curva, "var(--serie-1)", 2.5),
    "Curva sigmoide de la regresion logistica con el umbral de decision en 0.5");
  cifras("cifras-sigmoide", [
    ["Umbral (P = 0.5) en x", d.umbral], ["Peso w", d.peso], ["Sesgo b", d.sesgo],
    ["Exactitud", d.exactitud],
  ]);
}

// --- Grafico 4: hiperplano y margenes ------------------------------------
{
  const d = F.hiperplano;
  const e = escalas([d.clase_0, d.clase_1], [d.y_min, d.y_max]);
  lienzo("svg-hiperplano",
    ejes(e, "log(1+Confirmed)", "log(1+Recovered)"),
    nube(e, d.clase_0, "var(--serie-4)", 2.6, 0.45)
    + nube(e, d.clase_1, "var(--serie-1)", 2.6, 0.45)
    + anillos(e, d.soportes, "var(--ink-2)")
    + trazo(e, d.margen_superior, "var(--ink-2)", 1.5, "5 4")
    + trazo(e, d.margen_inferior, "var(--ink-2)", 1.5, "5 4")
    + trazo(e, d.frontera, "var(--ink)", 2.5),
    "Frontera del SVM lineal con sus dos margenes y los vectores de soporte marcados");
  cifras("cifras-hiperplano", [
    ["Vectores de soporte", `${d.total_soportes} de ${d.muestras}`],
    ["Ancho del margen", d.ancho_margen], ["Exactitud", d.exactitud],
  ]);
}

// --- Veredicto -----------------------------------------------------------
const rc = DATOS.resumen.covid;
const rk = DATOS.resumen.customer;
const mejorCovid = DATOS.covid[0];
const mejorCustomer = DATOS.customer[0];

document.getElementById("ganador-covid").textContent = rc.mejor_modelo;
document.getElementById("cifra-covid").textContent =
  `R\u00b2 ${fmt(mejorCovid.R2)} · RMSE ${fmt(mejorCovid.RMSE)} · baseline ${fmt(rc.baseline_rmse)}`;
document.getElementById("ganador-customer").textContent = "Ninguno supera el baseline";
document.getElementById("cifra-customer").textContent =
  `Mejor modelo ${fmt(mejorCustomer.Exactitud)} frente a ${fmt(rk.baseline)} de responder siempre «${rk.clase_mayoritaria}»`;
document.getElementById("muestra-covid").textContent =
  DATOS.resumen.covid_muestra.toLocaleString("es");

// --- Grafico covid (con selector de metrica) -----------------------------
leyenda("leyenda-covid");
const pieCovid = document.getElementById("pie-covid");

function pintarCovid(metrica) {
  const max = metrica === "R2" ? 1 : Math.max(...DATOS.covid.map((r) => r[metrica])) * 1.1;
  filas("filas-covid", DATOS.covid, metrica, max, null);
  marcas("marcas-covid", max, metrica === "R2" ? 2 : 1);
  pieCovid.textContent = metrica === "R2"
    ? `Barra más larga es mejor. El orden de las filas se mantiene fijo por R² para poder comparar entre métricas.`
    : `Barra más corta es mejor: ${metrica} es error. El orden de las filas sigue siendo el de R².`;
}

document.getElementById("selector-covid").addEventListener("click", (e) => {
  const boton = e.target.closest("button");
  if (!boton) return;
  document.querySelectorAll("#selector-covid button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b === boton)));
  pintarCovid(boton.dataset.metrica);
});
pintarCovid("R2");

// --- Grafico customer ----------------------------------------------------
leyenda("leyenda-customer");
filas("filas-customer", DATOS.customer, "Exactitud", 1, rk.baseline);
marcas("marcas-customer", 1, 2);
document.getElementById("nota-umbral").textContent = `Baseline ${fmt(rk.baseline)}`;
document.getElementById("pie-customer").textContent =
  `Ninguna de las ocho barras cruza la línea. Con ${rk.filas_prueba} filas de prueba, Segment no guarda relación con la edad ni con la ubicación del cliente.`;

// --- Grafico de costo ----------------------------------------------------
leyenda("leyenda-costo");
const porTiempo = [...DATOS.covid].sort((a, b) => b.Segundos - a.Segundos);
const maxSeg = Math.max(...porTiempo.map((r) => r.Segundos)) * 1.1;
filas("filas-costo", porTiempo, "Segundos", maxSeg, null);
marcas("marcas-costo", maxSeg, 0);

// --- Trazabilidad de la limpieza -----------------------------------------
const L = DATOS.limpieza;

tabla("tabla-origen", L.origen, [
  { clave: "archivo", titulo: "Archivo" },
  { clave: "ruta", titulo: "Ruta" },
  { clave: "bytes", titulo: "Bytes", num: true, decimales: 0 },
  { clave: "modificado", titulo: "Modificado" },
  { clave: "sha256", titulo: "SHA-256" },
]);

const campo = (etiqueta, valor, encendido) =>
  `<div class="campo${encendido ? " encendido" : ""}"><span>${etiqueta}</span><b>${valor}</b></div>`;

document.getElementById("limpieza").innerHTML = L.datasets.map((d) => {
  const inicial = d.filas_iniciales;
  const tramos = d.etapas.map((e) => {
    const quedan = (e.filas_despues / inicial) * 100;
    const quitadas = ((e.filas_antes - e.filas_despues) / inicial) * 100;
    return `<div class="tramo">
      <b>${e.etapa}</b>
      <div class="barra-tramo"><i style="width:${quedan.toFixed(2)}%"></i><u style="left:${quedan.toFixed(2)}%;width:${quitadas.toFixed(2)}%"></u></div>
      <span class="conteo">${e.detectadas.toLocaleString("es")} detectadas &middot; ${e.filas_antes.toLocaleString("es")} &rarr; ${e.filas_despues.toLocaleString("es")}${
        e.filas_antes > e.filas_despues
          ? ` <b>(-${(e.filas_antes - e.filas_despues).toLocaleString("es")})</b>`
          : ""}</span>
    </div>`;
  }).join("");

  const sucia = d.etapas.find((e) => e.ejemplos.length);
  const ejemplo = sucia
    ? `<div><h4>Fila real descartada &mdash; ${sucia.etapa}</h4>${
        Object.entries(sucia.ejemplos[0]).map(([k, v]) => campo(k, v)).join("")}</div>`
    : "";

  const c = d.conversion;
  const antes = `<div><h4>Una fila antes de convertir</h4>${
    Object.entries(c.ejemplo_antes).map(([k, v]) => campo(k, v)).join("")}</div>`;
  const despues = `<div><h4>La misma fila, ya num&eacute;rica</h4>${
    c.activas_primera_fila.map((n) => campo(n, "1", true)).join("")}
    <span class="flecha">+ ${(c.columnas_generadas - c.activas_primera_fila.length).toLocaleString("es")} columnas m&aacute;s en 0</span></div>`;

  return `<div class="rastro">
    <h3>${d.dataset} <span>objetivo: ${d.objetivo}</span></h3>
    <div class="embudo">${tramos}</div>
    <div class="evidencia">${ejemplo}${antes}${despues}</div>
    <p class="pie-grafico">De ${inicial.toLocaleString("es")} filas y ${d.columnas_iniciales} columnas
    quedan ${d.etapas[d.etapas.length - 1].filas_despues.toLocaleString("es")} filas y
    ${c.columnas_finales.toLocaleString("es")} columnas num&eacute;ricas, repartidas en
    ${d.particion.entrenamiento.toLocaleString("es")} de entrenamiento y
    ${d.particion.prueba.toLocaleString("es")} de prueba.</p>
  </div>`;
}).join("");

document.getElementById("pie-corrida").textContent =
  `Generado desde los CSV el ${L.corrida} por src/dashboard.py`;

// --- Tablas --------------------------------------------------------------
tabla("tabla-covid", DATOS.covid, [
  { clave: "Familia", titulo: "Familia" },
  { clave: "Modelo", titulo: "Modelo" },
  { clave: "R2", titulo: "R²", num: true },
  { clave: "RMSE", titulo: "RMSE", num: true },
  { clave: "MAE", titulo: "MAE", num: true },
  { clave: "Segundos", titulo: "Segundos", num: true, decimales: 2 },
]);
tabla("tabla-enfoques", DATOS.enfoques, [
  { clave: "Enfoque", titulo: "Enfoque" },
  { clave: "Que cambia", titulo: "Qu\u00e9 cambia" },
  { clave: "Columnas", titulo: "Columnas", num: true, decimales: 0 },
  { clave: "Mejor modelo", titulo: "Mejor modelo" },
  { clave: "Exactitud", titulo: "Exactitud", num: true },
  { clave: "Baseline", titulo: "Baseline", num: true },
  { clave: "Supera", titulo: "\u00bfSupera?", estado: true },
], (r) => r.Supera === "si");

tabla("tabla-customer", DATOS.customer, [
  { clave: "Familia", titulo: "Familia" },
  { clave: "Modelo", titulo: "Modelo" },
  { clave: "Exactitud", titulo: "Exactitud", num: true },
  { clave: "F1 macro", titulo: "F1 macro", num: true },
  { clave: "Segundos", titulo: "Segundos", num: true, decimales: 2 },
]);
</script>
"""


def main():
    datos = cargar()
    html = PLANTILLA.replace('%%DATOS%%', json.dumps(datos, ensure_ascii=False))
    salida = RESULTADOS / 'dashboard.html'
    salida.write_text(html, encoding='utf-8')
    print(f"Dashboard escrito en {salida}")


if __name__ == '__main__':
    main()
