"""
Augmentation operators for Bi-Encoder training.
Applied on-the-fly during data loading; no augmented data is persisted to disk.

Operators:
    shuffle_blocks   — permuta aleatoriamente el orden de bloques [BLK_*] como unidades atómicas
    shuffle_columns  — permuta columnas [COL]/[VAL] dentro de cada bloque
    mask_attributes  — reemplaza valores de campos con NULL (simula datos faltantes)
    inject_typos     — introduce errores tipográficos en valores de texto
    delete_span      — elimina una secuencia contigua de tokens no-especiales
"""

import random
import re
from dataclasses import dataclass

# Patrones de parsing
_BLK_SPLIT   = re.compile(r'(\[BLK_\w+\])')
_COL_SPLIT   = re.compile(r'(?=\[COL\])')
_TOKENIZE    = re.compile(r'\[[^\]]+\]|\S+')
_SPECIAL_TOK = re.compile(r'^\[(?:BLK_\w+|COL|VAL)\]$')
_VAL_SEGMENT = re.compile(r'(\[VAL\] )(.*?)(?=\s*\[(?:COL\]|BLK_)|\Z)', re.DOTALL)

_TYPO_CHARS = 'abcdefghijklmnopqrstuvwxyzáéíóúüñ'


@dataclass
class AugmentationConfig:
    use_shuffle_blocks:  bool  = True
    use_shuffle_columns: bool  = True
    use_mask_attributes: bool  = True
    use_inject_typos:    bool  = True
    use_delete_span:     bool  = True
    shuffle_blocks_prob:  float = 0.30  # prob de aplicar shuffle de bloques al registro
    shuffle_columns_prob: float = 0.30  # prob de aplicar shuffle de columnas al registro
    mask_prob:   float = 0.25   # probabilidad por atributo de enmascarar su valor (convertirlo a NULL)
    typo_prob:   float = 0.25   # probabilidad por palabra de introducir un typo
    delete_prob: float = 0.0   # probabilidad de aplicar span deletion al registro
    # Campos que nunca deben ser enmascarados — son llaves del ground truth
    protected_fields: tuple = (
        "nombre", "NOMBRE_DEL_PACIENTE", "NOMBRE_COMPLETO",
        "expediente", "EXP", "EXPEDIENTE",
    )


def shuffle_blocks(text: str) -> str:
    """Permuta aleatoriamente el orden de bloques semánticos.

    Cada [BLK_*] y su contenido [COL]/[VAL] se mueven como unidad atómica —
    equivalente a permutar SERIALIZATION_ORDER en tiempo de entrenamiento.
    """
    parts = _BLK_SPLIT.split(text)
    prefix = parts[0]
    blocks = [parts[i] + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    random.shuffle(blocks)
    return (prefix + ''.join(blocks)).strip()


def shuffle_columns(text: str) -> str:
    """Permuta aleatoriamente las columnas [COL]/[VAL] dentro de cada bloque.

    Complementa shuffle_blocks: mientras ese permuta bloques completos, este
    permuta los pares [COL]...[VAL] dentro de cada bloque individualmente.
    """
    parts = _BLK_SPLIT.split(text)
    prefix = parts[0]
    out = prefix
    for i in range(1, len(parts) - 1, 2):
        blk_tag = parts[i]
        blk_content = parts[i + 1]
        cols = _COL_SPLIT.split(blk_content)
        pre = cols[0]           # texto antes del primer [COL] (usualmente vacío)
        col_pairs = cols[1:]    # cada elemento empieza con [COL]
        random.shuffle(col_pairs)
        out += blk_tag + pre + ''.join(col_pairs)
    return out.strip()


def mask_attributes(text: str, mask_prob: float = 0.15,
                    protected_fields: tuple = ()) -> str:
    """Reemplaza valores de campo con NULL con probabilidad mask_prob.

    Simula la dispersión de datos real entre los 3 CSVs del INER.
    Valores ya nulos (NULL) y campos en protected_fields se omiten.
    """
    # Construir set de campos protegidos para lookup O(1)
    protected = set(protected_fields)

    # Precalcular posición de cada [COL] para saber a qué campo pertenece cada [VAL]
    col_pattern = re.compile(r'\[COL\]\s+(\S.*?)(?=\s*\[(?:COL|VAL|BLK_)|\Z)', re.DOTALL)
    col_positions = {m.start(): m.group(1).strip() for m in col_pattern.finditer(text)}
    col_starts = sorted(col_positions)

    def _field_for_match(val_start: int) -> str:
        """Devuelve el nombre del campo [COL] que precede a este [VAL]."""
        preceding = [s for s in col_starts if s < val_start]
        return col_positions[preceding[-1]] if preceding else ""

    def _replace(m):
        val = m.group(2).rstrip()
        if val == 'NULL':
            return m.group(0)
        if _field_for_match(m.start()) in protected:
            return m.group(0)
        if random.random() >= mask_prob:
            return m.group(0)
        return m.group(1) + 'NULL' + m.group(2)[len(val):]

    return _VAL_SEGMENT.sub(_replace, text)


def _apply_typo(word: str) -> str:
    pos = random.randint(0, len(word) - 1)
    op = random.choice(('delete', 'insert', 'swap', 'replace'))
    if op == 'delete':
        return word[:pos] + word[pos + 1:]
    if op == 'insert':
        return word[:pos] + random.choice(_TYPO_CHARS) + word[pos:]
    if op == 'swap' and pos < len(word) - 1:
        lst = list(word)
        lst[pos], lst[pos + 1] = lst[pos + 1], lst[pos]
        return ''.join(lst)
    # replace
    return word[:pos] + random.choice(_TYPO_CHARS) + word[pos + 1:]


def inject_typos(text: str, typo_prob: float = 0.05) -> str:
    """Introduce errores tipográficos en valores de texto.

    Solo afecta tokens de valor reales — no tokens especiales ni valores NULL.
    Modela los errores de captura manual típicos del entorno clínico del INER.
    """
    def _replace(m):
        val = m.group(2).rstrip()
        if val == 'NULL':
            return m.group(0)
        words = val.split()
        new_words = [
            _apply_typo(w) if random.random() < typo_prob and len(w) > 1 else w
            for w in words
        ]
        return m.group(1) + ' '.join(new_words) + m.group(2)[len(val):]
    return _VAL_SEGMENT.sub(_replace, text)


def delete_span(text: str, delete_prob: float = 0.0, max_span: int = 3) -> str:
    """Elimina una secuencia contigua de tokens no-especiales.

    Fuerza al modelo a no depender de palabras específicas y a usar contexto global.
    Los tokens especiales ([BLK_*], [COL], [VAL]) nunca se eliminan.
    """
    if random.random() >= delete_prob:
        return text
    tokens = _TOKENIZE.findall(text)
    non_special = [i for i, t in enumerate(tokens) if not _SPECIAL_TOK.match(t)]
    if not non_special:
        return text
    start = random.randint(0, len(non_special) - 1)
    span = min(random.randint(1, max_span), len(non_special) - start)
    to_delete = set(non_special[start:start + span])
    return ' '.join(t for i, t in enumerate(tokens) if i not in to_delete)


def augment(text: str, config: AugmentationConfig) -> str:
    """Aplica los operadores configurados a un registro serializado.

    Orden fijo: shuffle → mask → typos → delete.
    Cada operador se controla independientemente desde config.
    """
    if config.use_shuffle_blocks and random.random() < config.shuffle_blocks_prob:
        text = shuffle_blocks(text)
    if config.use_shuffle_columns and random.random() < config.shuffle_columns_prob:
        text = shuffle_columns(text)
    if config.use_mask_attributes:
        text = mask_attributes(text, config.mask_prob, config.protected_fields)
    if config.use_inject_typos:
        text = inject_typos(text, config.typo_prob)
    if config.use_delete_span:
        text = delete_span(text, config.delete_prob)
    return text
