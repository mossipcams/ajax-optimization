//! Byte-for-byte reimplementation of what
//! `yaml.dump(data, Dumper=yaml.CDumper, default_flow_style=False, allow_unicode=True,
//! sort_keys=False).replace(": null\n", ":\n")` emits for a list of flat str/None mappings.
//!
//! Scalar style choice follows PyYAML's representer + resolver (a str that would resolve to a
//! non-str tag when plain is quoted) and libyaml's `yaml_emitter_analyze_scalar` /
//! `yaml_emitter_select_scalar_style`. Plain and single-quoted writers follow
//! `yaml_emitter_write_plain` / `yaml_emitter_write_single_quoted`, including line folding at
//! best_width 80. Anything that would need the double-quoted writer or contains a line break
//! returns `None` so the caller falls back to PyYAML.

use once_cell::sync::Lazy;
use regex::RegexSet;

const BEST_WIDTH: usize = 80;
const VALUE_INDENT: usize = 4;

/// PyYAML resolver.py implicit resolvers (Resolver, YAML 1.1). Python's `re.match(...$)` also
/// matches before a single trailing "\n"; values with line breaks never reach these checks.
static IMPLICIT: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$",
        r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
        r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
        r"^(?:<<)$",
        r"^(?:~|null|Null|NULL|)$",
        r"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$",
        r"^(?:=)$",
    ])
    .unwrap()
});

/// Mirrors PyYAML's per-first-character resolver lookup: a value whose first character no
/// implicit resolver is registered for always resolves to str, so most names skip the regex set.
fn resolves_to_non_str(value: &str) -> bool {
    match value.as_bytes().first() {
        None => true, // "" -> null
        Some(b) if b"yYnNtTfFoO-+0123456789.<~=".contains(b) => {
            IMPLICIT.is_match(value)
        }
        Some(_) => false,
    }
}

fn is_break(c: char) -> bool {
    matches!(c, '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

fn is_blankz(c: Option<char>) -> bool {
    match c {
        None => true,
        Some(c) => c == ' ' || c == '\t' || is_break(c) || c == '\0',
    }
}

/// libyaml IS_PRINTABLE, expressed on code points.
fn is_printable(c: char) -> bool {
    let u = c as u32;
    u == 0x0A
        || (0x20..=0x7E).contains(&u)
        || (0xA0..=0xD7FF).contains(&u)
        || ((0xE000..=0xFFFD).contains(&u) && u != 0xFEFF)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Style {
    Plain,
    Single,
}

/// Returns None when the scalar needs a style this module does not write.
pub fn select_style(value: &str) -> Option<Style> {
    let chars: Vec<char> = value.chars().collect();
    let len = chars.len();
    let (mut block_plain, mut single) = (true, true);
    if len == 0 {
        // libyaml: block_plain_allowed=1, single_quoted_allowed=1; plain_implicit is false.
        return Some(Style::Single);
    }
    if value.starts_with("---") || value.starts_with("...") {
        block_plain = false;
    }
    let (mut leading_space, mut trailing_space) = (false, false);
    let (mut line_breaks, mut special) = (false, false);
    let mut previous_space = false;
    let mut preceded_by_ws = true;
    let mut followed_by_ws = is_blankz(chars.get(1).copied());
    for (i, &c) in chars.iter().enumerate() {
        if i == 0 {
            if "#,[]{}&*!|>'\"%@`".contains(c) {
                block_plain = false;
            }
            if (c == '?' || c == ':') && followed_by_ws {
                block_plain = false;
            }
            if c == '-' && followed_by_ws {
                block_plain = false;
            }
        } else {
            if c == ':' && followed_by_ws {
                block_plain = false;
            }
            if c == '#' && preceded_by_ws {
                block_plain = false;
            }
        }
        if !is_printable(c) {
            special = true;
        }
        if is_break(c) {
            line_breaks = true;
        }
        if c == ' ' {
            if i == 0 {
                leading_space = true;
            }
            if i == len - 1 {
                trailing_space = true;
            }
            previous_space = true;
        } else {
            let _ = previous_space;
            previous_space = false;
        }
        preceded_by_ws = is_blankz(Some(c));
        followed_by_ws = is_blankz(chars.get(i + 2).copied());
    }
    if special || line_breaks {
        return None; // double-quoted or break-aware writers: fall back to PyYAML
    }
    if leading_space || trailing_space {
        block_plain = false;
    }
    let _ = &mut single;
    if block_plain && !resolves_to_non_str(value) {
        return Some(Style::Plain);
    }
    if single {
        Some(Style::Single)
    } else {
        None
    }
}

struct Writer {
    out: String,
    column: usize,
}

impl Writer {
    fn put(&mut self, c: char) {
        self.out.push(c);
        self.column += 1;
    }
    fn put_str(&mut self, s: &str) {
        for c in s.chars() {
            self.put(c);
        }
    }
    fn indent(&mut self, indent: usize) {
        // yaml_emitter_write_indent mid-scalar: column > indent always holds here.
        self.out.push('\n');
        self.column = 0;
        for _ in 0..indent {
            self.put(' ');
        }
    }

    fn plain(&mut self, value: &str) {
        self.put(' ');
        let chars: Vec<char> = value.chars().collect();
        let mut spaces = false;
        for (i, &c) in chars.iter().enumerate() {
            if c == ' ' {
                if !spaces && self.column > BEST_WIDTH && chars.get(i + 1) != Some(&' ') {
                    self.indent(VALUE_INDENT);
                } else {
                    self.put(c);
                }
                spaces = true;
            } else {
                self.put(c);
                spaces = false;
            }
        }
    }

    fn single_quoted(&mut self, value: &str) {
        self.put(' ');
        self.put('\'');
        let chars: Vec<char> = value.chars().collect();
        let n = chars.len();
        let mut spaces = false;
        for (i, &c) in chars.iter().enumerate() {
            if c == ' ' {
                if !spaces
                    && self.column > BEST_WIDTH
                    && i != 0
                    && i != n - 1
                    && chars.get(i + 1) != Some(&' ')
                {
                    self.indent(VALUE_INDENT);
                } else {
                    self.put(c);
                }
                spaces = true;
            } else {
                if c == '\'' {
                    self.put('\'');
                }
                self.put(c);
                spaces = false;
            }
        }
        self.put('\'');
    }
}

/// Keys are written plain; they must be simple plain-safe identifiers.
fn key_ok(key: &str) -> bool {
    !key.is_empty()
        && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && select_style(key) == Some(Style::Plain)
}

pub fn dump_entries<S: AsRef<str>>(entries: &[Vec<(S, Option<S>)>]) -> Option<String> {
    if entries.is_empty() {
        return None; // PyYAML writes "[]\n"; not worth special-casing
    }
    let mut w = Writer { out: String::with_capacity(entries.len() * 64), column: 0 };
    for entry in entries {
        if entry.is_empty() {
            return None; // "- {}"
        }
        for (k, (key, value)) in entry.iter().enumerate() {
            let key = key.as_ref();
            if !key_ok(key) {
                return None;
            }
            w.put_str(if k == 0 { "- " } else { "  " });
            w.put_str(key);
            w.put(':');
            match value {
                None => {} // ": null\n" is rewritten to ":\n" by the caller's .replace
                Some(v) => match select_style(v.as_ref())? {
                    Style::Plain => w.plain(v.as_ref()),
                    Style::Single => w.single_quoted(v.as_ref()),
                },
            }
            w.out.push('\n');
            w.column = 0;
        }
    }
    Some(w.out)
}
