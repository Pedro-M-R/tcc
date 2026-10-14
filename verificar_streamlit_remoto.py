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


def campos_formulario(elementos, formulario, consultar=False):
    """Escolhe a ação pelo rótulo, sem depender da ordem dos botões."""
    rotulos = ({'Consultar checagens pelo texto'} if consultar else
               {'Analisar notícia', 'Analisar notícia com SVM'})
    if formulario == 'link' and not consultar:
        rotulos = {'Ler link e analisar'}
    campos = {}
    for delta in elementos:
        e = delta.get('newElement', {})
        for tipo in ('textInput', 'textArea', 'button'):
            campo = e.get(tipo)
            if not campo or campo.get('formId') != formulario:
                continue
            if tipo == 'button' and campo.get('label') not in rotulos:
                continue
            campos[tipo] = campo['id']
    esperados = {'textInput', 'button'} if formulario == 'link' else {'textInput', 'textArea', 'button'}
    if set(campos) != esperados:
        raise ValueError(f'O formulário {formulario} ou a ação solicitada não foi encontrado no app remoto.')
    return campos


def verificar(url, titulo=None, texto=None, link=None, consultar=False):
    if consultar and link is not None:
        raise ValueError('A consulta por texto não recebe link.')
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
            formulario = 'link' if link is not None else 'noticia'
            campos = campos_formulario(elementos, formulario, consultar)
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
