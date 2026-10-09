"""Verifica resultados reais do app público sem acessar o painel privado."""

import json
from pathlib import Path

from verificar_streamlit_remoto import verificar


def main():
    registros = []
    casos = [('opiniao_bonita', {'titulo': 'pedro eh bonita', 'texto': 'pedro eh bonita'}),
             ('opiniao_bonito', {'titulo': 'pedro eh bonito', 'texto': 'pedro eh bonito'}),
             ('noticia_curta', {'texto': 'O prefeito anunciou uma obra.'})]
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
        inconclusivo = any(e.get('alert', {}).get('body') == 'Análise inconclusiva'
                          for e in elementos)
        registro = {'caso': nome, 'resultados': titulos, 'rodape': rodape,
                    'erros': erros, 'inconclusivo': inconclusivo,
                    'quantidade_escores': len(metricas)}
        registros.append(registro)
        print(json.dumps(registro, ensure_ascii=True), flush=True)
        assert not erros and not inconclusivo, registro
        assert any('2026.10.09-checagem1' in s for s in rodape), registro
        if nome.startswith('checagem'):
            assert len(titulos) == 1 and 'Há indícios de que seja uma alegação falsa' in titulos[0], registro
            assert 'Boatos.org' in titulos[0] and not metricas, registro
            assert not any('Há indícios de que seja verdadeira' in s for s in mensagens), registro
        elif nome.startswith('opiniao'):
            assert any('<h2>Falsa</h2>' in s for s in titulos), registro
            assert not metricas, registro
        else:
            assert titulos and len(metricas) == 2, registro
    saida = Path('resultados/revisao_casos_reais/publicacao_checagem1.json')
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Publicacao validada: {len(casos)} casos, incluindo os dois links nos dois formularios.')


if __name__ == '__main__':
    main()
