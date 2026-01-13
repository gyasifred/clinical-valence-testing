#!/usr/bin/env python3
"""Convert Markdown to PDF for project overview using reportlab."""

import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def create_styles():
    """Create custom styles for the PDF."""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.HexColor('#003366'),
        alignment=TA_CENTER
    ))

    # Section headers
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#004C99'),
        borderWidth=1,
        borderColor=colors.HexColor('#004C99'),
        borderPadding=5
    ))

    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#333333')
    ))

    styles.add(ParagraphStyle(
        name='SubsubsectionHeader',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#444444')
    ))

    # Body text
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))

    # Code style
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.HexColor('#F0F0F0'),
        borderWidth=1,
        borderColor=colors.HexColor('#CCCCCC'),
        borderPadding=5,
        spaceBefore=5,
        spaceAfter=5
    ))

    # Bullet style
    styles.add(ParagraphStyle(
        name='BulletItem',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceBefore=2,
        spaceAfter=2,
        bulletIndent=10
    ))

    return styles


def clean_markdown_text(text):
    """Remove markdown formatting from text."""
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Code inline
    text = re.sub(r'`(.*?)`', r'<font name="Courier" size="9">\1</font>', text)
    # Links - keep text only
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text


def parse_markdown_to_elements(md_content, styles):
    """Parse markdown content and return reportlab elements."""
    elements = []
    lines = md_content.split('\n')

    i = 0
    in_code_block = False
    code_buffer = []
    in_table = False
    table_data = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                code_text = '<br/>'.join(code_buffer)
                elements.append(Paragraph(code_text, styles['CodeBlock']))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells and not all(set(c.strip()) <= {'-', ':'} for c in cells):
                if not in_table:
                    in_table = True
                    table_data = []
                table_data.append([clean_markdown_text(c) for c in cells])
            i += 1
            continue
        elif in_table:
            # End of table - create table element
            if table_data:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDDDDD')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(t)
                elements.append(Spacer(1, 10))
            in_table = False
            table_data = []

        # Main title (# )
        if line.startswith('# '):
            text = clean_markdown_text(line[2:].strip())
            elements.append(Paragraph(text, styles['MainTitle']))
            elements.append(Spacer(1, 10))

        # Section header (## )
        elif line.startswith('## '):
            text = clean_markdown_text(line[3:].strip())
            elements.append(Spacer(1, 5))
            elements.append(Paragraph(text, styles['SectionHeader']))

        # Subsection header (### )
        elif line.startswith('### '):
            text = clean_markdown_text(line[4:].strip())
            elements.append(Paragraph(text, styles['SubsectionHeader']))

        # Subsubsection header (#### )
        elif line.startswith('#### '):
            text = clean_markdown_text(line[5:].strip())
            elements.append(Paragraph(text, styles['SubsubsectionHeader']))

        # Horizontal rule
        elif line.strip() in ('---', '***', '___'):
            elements.append(Spacer(1, 10))

        # Bullet points
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            text = clean_markdown_text(line.strip()[2:])
            bullet_text = f"&bull; {text}"
            elements.append(Paragraph(bullet_text, styles['BulletItem']))

        # Numbered lists
        elif re.match(r'^\s*\d+\.\s', line):
            text = re.sub(r'^\s*\d+\.\s', '', line)
            text = clean_markdown_text(text)
            # Get number
            num_match = re.match(r'^\s*(\d+)\.', line)
            num = num_match.group(1) if num_match else '1'
            bullet_text = f"{num}. {text}"
            elements.append(Paragraph(bullet_text, styles['BulletItem']))

        # Regular paragraph
        elif line.strip():
            text = clean_markdown_text(line.strip())
            elements.append(Paragraph(text, styles['CustomBody']))

        # Empty line
        else:
            elements.append(Spacer(1, 6))

        i += 1

    # Handle remaining table
    if in_table and table_data:
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDDDDD')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(t)

    return elements


def markdown_to_pdf(md_file: str, pdf_file: str):
    """Convert markdown file to PDF."""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Create styles
    styles = create_styles()

    # Parse markdown and create elements
    elements = parse_markdown_to_elements(md_content, styles)

    # Build PDF
    doc.build(elements)
    print(f"PDF generated: {pdf_file}")


if __name__ == "__main__":
    md_file = Path(__file__).parent / "PROJECT_OVERVIEW.md"
    pdf_file = Path(__file__).parent / "PROJECT_OVERVIEW.pdf"
    markdown_to_pdf(str(md_file), str(pdf_file))
