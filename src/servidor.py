"""
Servidor local de la aplicacion: entrenar los modelos y ver los resultados en una sola ventana.

Sirve src/app.html (menu lateral con tres vistas: Entrenar, Ocho modelos y Red
neuronal) y los archivos de resultados/. La vista Entrenar corre los scripts del
proyecto en un hilo aparte y transmite su salida en vivo (Server-Sent Events);
si se recarga la pagina, la corrida sigue y el registro se vuelve a mostrar.

Solo escucha en 127.0.0.1 y solo ejecuta los scripts de la lista PASOS.

Uso:  npm run dev
      .venv/bin/python src/servidor.py        (lo mismo, sin npm)
"""

import json
import mimetypes
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / 'resultados'
APP = Path(__file__).resolve().parent / 'app.html'
PUERTOS = range(8000, 8011)

# Orden de ejecucion de "Entrenar todo". 'segundos' es la duracion medida en la
# corrida de referencia, solo para que la interfaz muestre una estimacion.
# 'registro': archivo de resultados/ donde se guarda lo que imprimio el script
# (esos scripts no escriben su propio .txt; antes se generaba a mano con tee).
PASOS = [
    {'id': 'supervisado', 'nombre': 'Ocho modelos sobre los dos datasets',
     'script': 'aprendizaje_supervisado.py', 'args': [], 'segundos': 200, 'registro': 'salida_completa.txt',
     'salidas': ['resultados_covid.csv', 'resultados_customer.csv', 'resumen.json', 'limpieza.json']},
    {'id': 'diagnostico', 'nombre': 'Diagnóstico de Customer.csv',
     'script': 'diagnostico_customer.py', 'args': [], 'segundos': 60, 'registro': 'diagnostico_customer.txt',
     'salidas': ['diagnostico_customer.txt']},
    {'id': 'enfoques', 'nombre': 'Ocho formas de preparar Customer.csv',
     'script': 'enfoques_customer.py', 'args': [], 'segundos': 20, 'registro': 'enfoques_customer.txt',
     'salidas': ['enfoques_customer.csv', 'enfoques_customer.txt']},
    {'id': 'funciones', 'nombre': 'Funciones de los apuntes',
     'script': 'funciones.py', 'args': [], 'segundos': 10,
     'salidas': ['funciones.json']},
    {'id': 'red', 'nombre': 'Red neuronal desde cero',
     'script': 'red_neuronal.py', 'args': ['--sin-preguntas'], 'segundos': 15,
     'salidas': ['red_neuronal.html', 'red_neuronal.json']},
    {'id': 'dashboard', 'nombre': 'Dashboard de los ocho modelos',
     'script': 'dashboard.py', 'args': [], 'segundos': 2,
     'salidas': ['dashboard.html']},
]
POR_ID = {p['id']: p for p in PASOS}


class Corrida:
    """Una ejecucion de uno o varios pasos. Guarda todos sus eventos para poder reenviarlos."""

    def __init__(self, ids):
        self.ids = ids
        self.eventos = []
        self.terminada = False
        self.cancelada = False
        self.proceso = None
        self.cond = threading.Condition()

    def emitir(self, **evento):
        with self.cond:
            self.eventos.append({**evento, 't': round(time.time(), 2)})
            self.cond.notify_all()

    def ejecutar(self):
        ok = True
        self.emitir(tipo='corrida', pasos=self.ids, hora=time.strftime('%H:%M:%S'))
        for paso_id in self.ids:
            if self.cancelada:
                break
            paso = POR_ID[paso_id]
            self.emitir(tipo='inicio', paso=paso_id)
            inicio = time.monotonic()
            lineas = []
            try:
                self.proceso = subprocess.Popen(
                    [sys.executable, '-u', str(RAIZ / 'src' / paso['script']), *paso['args']],
                    cwd=RAIZ, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for linea in self.proceso.stdout:
                    lineas.append(linea)
                    self.emitir(tipo='linea', paso=paso_id, texto=linea.rstrip('\n'))
                codigo = self.proceso.wait()
                if codigo == 0 and 'registro' in paso:
                    (RESULTADOS / paso['registro']).write_text(''.join(lineas), encoding='utf-8')
            except OSError as error:
                self.emitir(tipo='linea', paso=paso_id, texto=f'No se pudo ejecutar: {error}')
                codigo = -1
            self.emitir(tipo='fin', paso=paso_id, codigo=codigo,
                        segundos=round(time.monotonic() - inicio, 1))
            if codigo != 0:
                ok = False
                break
        self.emitir(tipo='terminado', ok=ok and not self.cancelada, cancelada=self.cancelada)
        with self.cond:
            self.terminada = True
            self.cond.notify_all()

    def cancelar(self):
        self.cancelada = True
        if self.proceso and self.proceso.poll() is None:
            self.proceso.terminate()


corrida = None                  # la ultima corrida, terminada o en curso
cerrojo = threading.Lock()      # impide lanzar dos corridas a la vez


def estado_salidas():
    """Fecha de la ultima modificacion de la salida principal de cada paso."""
    estado = {}
    for paso in PASOS:
        ruta = RESULTADOS / paso['salidas'][0]
        estado[paso['id']] = (time.strftime('%Y-%m-%d %H:%M', time.localtime(ruta.stat().st_mtime))
                              if ruta.exists() else None)
    return estado


class Manejador(BaseHTTPRequestHandler):
    server_version = 'MLServidor'

    def log_message(self, formato, *args):
        pass  # la consola queda para la URL y los errores

    def host_valido(self):
        # Rechaza peticiones con otro Host (ataques de DNS rebinding desde paginas externas).
        puerto = self.server.server_address[1]
        return self.headers.get('Host') in (f'127.0.0.1:{puerto}', f'localhost:{puerto}')

    def responder_json(self, datos, estado=HTTPStatus.OK):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode('utf-8')
        self.send_response(estado)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(cuerpo)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(cuerpo)

    def enviar_archivo(self, ruta):
        if not ruta.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, 'No existe; corre el entrenamiento para generarlo.')
            return
        cuerpo = ruta.read_bytes()
        tipo = mimetypes.guess_type(ruta.name)[0] or 'application/octet-stream'
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', f'{tipo}; charset=utf-8' if tipo.startswith('text/') else tipo)
        self.send_header('Content-Length', str(len(cuerpo)))
        self.send_header('Cache-Control', 'no-store')  # tras entrenar, las vistas deben ver lo nuevo
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if not self.host_valido():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        if url.path in ('/', '/index.html'):
            self.enviar_archivo(APP)
        elif url.path.startswith('/resultados/'):
            ruta = (RESULTADOS / url.path.removeprefix('/resultados/')).resolve()
            if ruta.parent != RESULTADOS:  # nada fuera de resultados/, ni subcarpetas
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.enviar_archivo(ruta)
        elif url.path == '/api/estado':
            self.responder_json({
                'pasos': [{k: p[k] for k in ('id', 'nombre', 'script', 'segundos', 'salidas')} for p in PASOS],
                'salidas': estado_salidas(),
                'corriendo': bool(corrida and not corrida.terminada),
                'hay_corrida': corrida is not None,
            })
        elif url.path == '/api/eventos':
            self.transmitir(url)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def transmitir(self, url):
        """Server-Sent Events con los eventos de la corrida, desde el indice pedido."""
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        c = corrida
        if c is None:
            self.wfile.write(b'data: {"tipo": "sin_corrida"}\n\n')
            return
        ultimo = self.headers.get('Last-Event-ID')
        indice = int(ultimo) + 1 if ultimo and ultimo.isdigit() else int(
            parse_qs(url.query).get('desde', ['0'])[0] or 0)
        try:
            while True:
                with c.cond:
                    if indice >= len(c.eventos) and not c.terminada:
                        c.cond.wait(timeout=15)
                    nuevos, terminada = c.eventos[indice:], c.terminada
                if not nuevos and not terminada:
                    self.wfile.write(b': sigue\n\n')  # mantiene viva la conexion
                for evento in nuevos:
                    datos = json.dumps(evento, ensure_ascii=False)
                    self.wfile.write(f'id: {indice}\ndata: {datos}\n\n'.encode('utf-8'))
                    indice += 1
                self.wfile.flush()
                if terminada and indice >= len(c.eventos):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return  # la pestana se cerro; la corrida sigue en su hilo

    def do_POST(self):
        global corrida
        if not self.host_valido():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        if url.path == '/api/correr':
            try:
                largo = int(self.headers.get('Content-Length', 0))
                ids = json.loads(self.rfile.read(largo) or b'{}').get('pasos') or []
            except (ValueError, AttributeError):
                self.responder_json({'error': 'Cuerpo invalido.'}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(ids, list) or not ids or any(i not in POR_ID for i in ids):
                self.responder_json({'error': 'Pasos desconocidos.'}, HTTPStatus.BAD_REQUEST)
                return
            with cerrojo:
                if corrida and not corrida.terminada:
                    self.responder_json({'error': 'Ya hay un entrenamiento en curso.'}, HTTPStatus.CONFLICT)
                    return
                orden = [p['id'] for p in PASOS if p['id'] in ids]  # siempre en el orden del pipeline
                corrida = Corrida(orden)
                threading.Thread(target=corrida.ejecutar, daemon=True).start()
            self.responder_json({'ok': True, 'pasos': orden}, HTTPStatus.ACCEPTED)
        elif url.path == '/api/cancelar':
            if corrida and not corrida.terminada:
                corrida.cancelar()
            self.responder_json({'ok': True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    servidor = None
    for puerto in PUERTOS:
        try:
            servidor = ThreadingHTTPServer(('127.0.0.1', puerto), Manejador)
            break
        except OSError:
            continue
    if servidor is None:
        sys.exit(f'Los puertos {PUERTOS.start}-{PUERTOS.stop - 1} estan ocupados.')
    servidor.daemon_threads = True
    url = f'http://127.0.0.1:{servidor.server_address[1]}/'
    print(f'Aplicacion en {url}  (Ctrl+C para detener)', flush=True)
    if '--sin-navegador' not in sys.argv:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        if corrida and not corrida.terminada:
            corrida.cancelar()
        print('\nServidor detenido.')


if __name__ == '__main__':
    main()
