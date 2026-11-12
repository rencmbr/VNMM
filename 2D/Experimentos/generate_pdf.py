"""
Script para conversão do artigo Markdown (Artigo_VNMM_Multimateriais.md) em PDF
utilizando estilização acadêmica (padrão periódico científico / Elsevier / IEEE),
renderização vetorial de fórmulas matemáticas (KaTeX) e incorporação de imagens em Base64.
"""

import os
import re
import sys
import base64
import subprocess
from pathlib import Path

DIR_EXP = Path(__file__).resolve().parent
MD_PATH = DIR_EXP / "Artigo_VNMM_Multimateriais.md"
PDF_PATH = DIR_EXP / "Artigo_VNMM_Multimateriais.pdf"
HTML_PATH = DIR_EXP / "Artigo_VNMM_Multimateriais.html"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def image_to_base64(img_path):
    """Converte imagem local para data URI em Base64 para incorporação perfeita."""
    if not os.path.isabs(img_path):
        full_path = DIR_EXP / img_path
    else:
        full_path = Path(img_path)
        
    if full_path.exists():
        with open(full_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    return img_path


def parse_markdown_to_html(md_text):
    """Converte markdown estruturado em HTML com suporte a tabelas, imagens e LaTeX."""
    lines = md_text.splitlines()
    html_lines = []
    
    in_table = False
    table_lines = []
    in_quote = False
    quote_lines = []
    
    def flush_table(tbl):
        if not tbl:
            return ""
        html = ["<div class='table-container'><table>"]
        header_done = False
        for line in tbl:
            if re.match(r'^\s*\|?\s*[-:]+[-| :]*$', line):
                header_done = True
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            tag = "th" if not header_done else "td"
            row = "".join([f"<{tag}>{c}</{tag}>" for c in cells])
            html.append(f"<tr>{row}</tr>")
        html.append("</table></div>")
        return "\n".join(html)

    def flush_quote(q):
        if not q:
            return ""
        content = " ".join(q)
        return f"<blockquote>{content}</blockquote>"

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Bloco de tabela
        if line.strip().startswith('|'):
            if in_quote:
                html_lines.append(flush_quote(quote_lines))
                quote_lines = []
                in_quote = False
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            html_lines.append(flush_table(table_lines))
            table_lines = []
            in_table = False
            
        # Bloco de citação
        if line.strip().startswith('>'):
            in_quote = True
            quote_lines.append(line.strip().lstrip('>').strip())
            i += 1
            continue
        elif in_quote:
            html_lines.append(flush_quote(quote_lines))
            quote_lines = []
            in_quote = False

        # Títulos
        if line.startswith('# '):
            html_lines.append(f"<h1 class='paper-title'>{line[2:].strip()}</h1>")
        elif line.startswith('## '):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith('### '):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.strip() == '---':
            html_lines.append("<hr/>")
        # Imagens
        elif re.match(r'!\[(.*?)\]\((.*?)\)', line.strip()):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
            alt, src = m.group(1), m.group(2)
            b64_src = image_to_base64(src)
            html_lines.append(
                f"<div class='figure-container'>"
                f"<img src='{b64_src}' alt='{alt}' />"
                f"</div>"
            )
        # Legenda de figura em itálico
        elif line.strip().startswith('*Figura') and line.strip().endswith('*'):
            caption = line.strip().strip('*')
            html_lines.append(f"<p class='figure-caption'>{caption}</p>")
        # Listas não ordenadas
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            item = line.strip()[2:].strip()
            html_lines.append(f"<ul><li>{item}</li></ul>")
        # Listas ordenadas
        elif re.match(r'^\d+\.\s', line.strip()):
            item = re.sub(r'^\d+\.\s', '', line.strip())
            html_lines.append(f"<ol><li>{item}</li></ol>")
        # Parágrafos normais
        elif line.strip():
            html_lines.append(f"<p>{line.strip()}</p>")
        else:
            html_lines.append("")
            
        i += 1

    if in_table:
        html_lines.append(flush_table(table_lines))
    if in_quote:
        html_lines.append(flush_quote(quote_lines))

    raw_html = "\n".join(html_lines)
    
    # Substituições inline
    # Negrito
    raw_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', raw_html)
    # Links
    raw_html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', raw_html)
    # Fusão de listas consecutivas
    raw_html = raw_html.replace("</ul>\n<ul>", "\n").replace("</ol>\n<ol>", "\n")
    
    return raw_html


def build_full_html(body_html):
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Interpolação e Modos Ressonantes em Meios Multimateriais - VNMM 2D</title>
    <!-- KaTeX para renderização de fórmulas matemáticas -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ],
            throwOnError: false
        }}); window.mathReady = true;"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 18mm 20mm 18mm;
            @bottom-center {{
                content: counter(page);
                font-family: 'Times New Roman', serif;
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: 'Times New Roman', 'Liberation Serif', Georgia, serif;
            font-size: 11pt;
            line-height: 1.55;
            color: #111;
            margin: 0;
            padding: 0;
            text-align: justify;
        }}
        .paper-title {{
            font-size: 19pt;
            font-weight: bold;
            text-align: center;
            line-height: 1.3;
            margin-top: 10px;
            margin-bottom: 18px;
            color: #0d1b2a;
        }}
        h2 {{
            font-size: 13.5pt;
            font-weight: bold;
            color: #1b263b;
            border-bottom: 1.2px solid #33415c;
            padding-bottom: 4px;
            margin-top: 24px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11.5pt;
            font-weight: bold;
            color: #2b3a4a;
            margin-top: 16px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 10px;
            text-indent: 1.5em;
        }}
        p.paper-meta {{
            text-align: center;
            text-indent: 0;
            font-size: 10pt;
            color: #333;
            margin-bottom: 18px;
        }}
        p.figure-caption {{
            text-align: center;
            text-indent: 0;
            font-size: 9.5pt;
            font-style: italic;
            color: #444;
            margin-top: 6px;
            margin-bottom: 18px;
            page-break-before: avoid;
        }}
        blockquote {{
            margin: 14px 20px;
            padding: 10px 16px;
            background-color: #f4f6f8;
            border-left: 3.5px solid #1b4965;
            font-size: 10pt;
            font-style: italic;
            border-radius: 2px;
        }}
        .figure-container {{
            text-align: center;
            margin-top: 16px;
            margin-bottom: 4px;
            page-break-inside: avoid;
        }}
        .figure-container img {{
            max-width: 96%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        }}
        .table-container {{
            margin: 16px 0;
            page-break-inside: avoid;
            display: flex;
            justify-content: center;
        }}
        table {{
            width: 98%;
            border-collapse: collapse;
            font-size: 9.5pt;
            margin: 0 auto;
        }}
        th, td {{
            padding: 6px 10px;
            text-align: center;
            border-top: 1px solid #ccc;
            border-bottom: 1px solid #ccc;
        }}
        th {{
            border-top: 2px solid #111;
            border-bottom: 1.5px solid #111;
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:last-child td {{
            border-bottom: 2px solid #111;
        }}
        ul, ol {{
            margin-top: 0;
            margin-bottom: 10px;
            padding-left: 30px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #ddd;
            margin: 20px 0;
        }}
        .katex-display {{
            margin: 12px 0 !important;
            text-align: center;
        }}
        code {{
            font-family: Consolas, 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f1f3f5;
            padding: 1px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""


def generate_pdf():
    print(f"Lendo markdown: {MD_PATH}...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    print("Convertendo markdown para HTML acadêmico com suporte a KaTeX...")
    body_html = parse_markdown_to_html(md_content)
    full_html = build_full_html(body_html)
    
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML intermediário salvo em: {HTML_PATH}")
    
    print("Renderizando PDF via Google Chrome Headless com KaTeX...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--virtual-time-budget=5000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if PDF_PATH.exists():
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"PDF gerado com sucesso!")
        print(f"Arquivo: {PDF_PATH}")
        print(f"Tamanho: {size_kb:.1f} KB")
    else:
        print("Erro ao gerar PDF:", res.stderr)


if __name__ == "__main__":
    generate_pdf()
