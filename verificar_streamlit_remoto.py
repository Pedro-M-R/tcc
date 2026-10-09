"""Verifica a execução de um app Streamlit público pelo protocolo do próprio app."""
import json
import sys
import time
from urllib.parse import urlsplit
import requests
from google.protobuf.json_format import MessageToDict
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.sync.client import connect


def receber(ws):
    elementos = []
    limite = time.monotonic() + 50
    while time.monotonic() < limite:
        msg = ForwardMsg()
        msg.ParseFromString(ws.recv(timeout=20))
        tipo = msg.WhichOneof('type')
        if tipo == 'delta':
            elementos.append(MessageToDict(msg.delta))
        if tipo == 'script_finished':
            return elementos
    raise TimeoutError('O Streamlit não concluiu a execução.')


def verificar(url, titulo=None, texto=None, link=None):
    wsurl = url.rstrip('/').replace('https://', 'wss://').replace('http://', 'ws://') + '/_stcore/stream'
    sessao = requests.Session()
    sessao.get(url, timeout=25).raise_for_status()
    pedido_http = sessao.prepare_request(requests.Request('GET', url.rstrip('/') + '/_stcore/stream'))
    headers = {'Cookie': pedido_http.headers.get('Cookie', '')}
    partes = urlsplit(url)
    origem = f'{partes.scheme}://{partes.netloc}'
    with connect(wsurl, origin=origem, subprotocols=['streamlit'], additional_headers=headers, open_timeout=20) as ws:
        pedido = BackMsg()
        pedido.rerun_script.query_string = ''
        ws.send(pedido.SerializeToString())
        elementos = receber(ws)
        radio_id = None
        radio_valor = None
        if link is not None:
            radios = [d.get('newElement', {}).get('radio', {}) for d in elementos]
            radio = next(r for r in radios if 'Usar um link' in r.get('options', []))
            radio_id = radio['id']
            radio_valor = 'Usar um link'
            pedido = BackMsg()
            widget = pedido.rerun_script.widget_states.widgets.add()
            widget.id = radio_id
            widget.string_value = radio_valor
            ws.send(pedido.SerializeToString())
            elementos = receber(ws)
        if titulo is not None or texto is not None or link is not None:
            campos = {}
            formulario = 'link' if link is not None else 'noticia'
            for delta in elementos:
                e = delta.get('newElement', {})
                for tipo in ('textInput', 'textArea', 'button'):
                    if tipo in e and e[tipo].get('formId') == formulario:
                        campos[tipo] = e[tipo]['id']
            esperados = {'textInput', 'button'} if link is not None else {'textInput', 'textArea', 'button'}
            if set(campos) != esperados:
                raise ValueError(f'O formulário {formulario} não foi encontrado no app remoto.')
            pedido = BackMsg()
            pedido.rerun_script.query_string = ''
            if radio_id is not None:
                widget = pedido.rerun_script.widget_states.widgets.add()
                widget.id = radio_id
                widget.string_value = radio_valor
            valores = [('textInput', link)] if link is not None else [('textInput', titulo or ''), ('textArea', texto or '')]
            for tipo, valor in valores:
                widget = pedido.rerun_script.widget_states.widgets.add()
                widget.id = campos[tipo]
                widget.string_value = valor
            botao = pedido.rerun_script.widget_states.widgets.add()
            botao.id = campos['button']
            botao.trigger_value = True
            ws.send(pedido.SerializeToString())
            elementos = receber(ws)
    return elementos


if __name__ == '__main__':
    print(json.dumps(verificar(sys.argv[1]), ensure_ascii=True, indent=2))
