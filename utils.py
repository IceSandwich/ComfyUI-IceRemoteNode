import re
from datetime import datetime

# Java SimpleDateFormat 常见符号 → Python strftime
_JAVA_TO_STRFTIME = [
    ('yyyy', '%Y'), ('yy',   '%y'),
    ('MMMM', '%B'), ('MMM',  '%b'), ('MM', '%m'),
    ('dd',   '%d'),
    ('HH',   '%H'), ('mm',   '%M'), ('ss', '%S'),
    ('SSS',  '%f'),        # 毫秒（需特判，见下）
    ('EEEE', '%A'), ('EEE',  '%a'),
    ('a',    '%p'),
    ('XXX',  '%z'), ('Z',    '%z'),
]

_DATE_RE = re.compile(r'%date:([^%]+)%')


def _java_to_strftime(fmt: str) -> str:
    """把 Java 日期格式字符串转换为 Python strftime 格式串。"""
    # 按长度倒序匹配，避免 "yyyy" 里的 "yy" 先被替换
    tokens = sorted(_JAVA_TO_STRFTIME, key=lambda x: -len(x[0]))
    pattern = re.compile('|'.join(re.escape(t) for t, _ in tokens))
    mapping = dict(_JAVA_TO_STRFTIME)
    return pattern.sub(lambda m: mapping[m.group(0)], fmt)


def format_template(template: str, dt: datetime | None = None) -> str:
    """替换模板里所有 %date:<java-format>% 占位符。"""
    if dt is None:
        dt = datetime.now()

    def _replace(match: re.Match) -> str:
        java_fmt = match.group(1)

        # SSS 是毫秒，Python 的 %f 是微秒，需要单独处理
        if 'SSS' in java_fmt:
            placeholder = '\x00MS\x00'
            py_fmt = _java_to_strftime(java_fmt.replace('SSS', placeholder))
            s = dt.strftime(py_fmt)
            return s.replace(placeholder, f'{dt.microsecond // 1000:03d}')

        return dt.strftime(_java_to_strftime(java_fmt))

    return _DATE_RE.sub(_replace, template)
