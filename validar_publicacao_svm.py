"""Verifica resultados reais do app público sem acessar o painel privado."""

import json
from pathlib import Path

from verificar_streamlit_remoto import verificar


def main():
    registros = []
    casos = [('consulta_insuficiente', {'texto': 'oi'}),
             ('checagem_por_texto', {'texto':
                 'Flávio Bolsonaro estaria planejando forjar um atentado em Salvador e simular '
                 'uma piora no estado de saúde de Jair Bolsonaro para influenciar o resultado '
                 'das eleições de 2026.'})]
    urls = {
        'eduarda': 'https://www.boatos.org/politica/e-falso-que-eduarda-campopiano-tenha-declarado-ser-contra-o-voto-feminino.html',
        'tse': 'https://www.boatos.org/politica/tse-vai-investigar-se-atraso-na-divulgacao-de-resultados-das-eleicoes-tem-relacao-com-fraude-contra-flavio-bolsonaro.html',
    }
    for nome, url in urls.items():
        casos.extend([(f'checagem_{nome}_link', {'link': url}),
                      (f'checagem_{nome}_texto', {'texto': url})])
    for nome, entrada in casos:
        deltas = verificar('https://tccpedro.streamlit.app/~/+/', **entrada)
        elementos = [d.get('newElement', {}) for d in deltas]
        mensagens = [e.get('markdown', {}).get('body', '') for e in elementos]
        erros = [e for e in elementos if 'exception' in e or
                 e.get('alert', {}).get('format') == 'ERROR']
        titulos = [s for s in mensagens if '<h2>' in s and not s.startswith('<style>')]
        rodape = [s for s in mensagens if 'class="rodape"' in s]
        metricas = [e['metric'] for e in elementos if 'metric' in e]
        inconclusivo = any(e.get('alert', {}).get('body') == 'Consulta às fontes inconclusiva'
                          for e in elementos)
        registro = {'caso': nome, 'resultados': titulos, 'rodape': rodape,
                    'erros': erros, 'inconclusivo': inconclusivo,
                    'quantidade_escores': len(metricas)}
        registros.append(registro)
        print(json.dumps(registro, ensure_ascii=True), flush=True)
        assert not erros, registro
        assert inconclusivo == (nome == 'consulta_insuficiente'), registro
        assert any('2026.10.14-consulta-unica' in s for s in rodape), registro
        if nome == 'consulta_insuficiente':
            assert not titulos and not metricas, registro
        elif nome.startswith('checagem'):
            assert len(titulos) == 1 and 'Há indícios de que seja uma alegação falsa' in titulos[0], registro
            assert 'Boatos.org' in titulos[0] and not metricas, registro
            assert not any('Há indícios de que seja verdadeira' in s for s in mensagens), registro
    saida = Path('resultados/revisao_textos_20261014/publicacao_consulta_unica.json')
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Publicacao validada: {len(casos)} casos, incluindo os dois links nos dois formularios.')


if __name__ == '__main__':
    main()
