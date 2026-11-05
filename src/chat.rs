//! Prompts from the model's own chat template, tokenization, and parsing of the model's output into reasoning,
//! content and tool calls, with Qwen3.5's details: Python's `str()` in the template, the XML tool format of
//! Qwen3-Coder with typed parameters, two end tokens, a think block that ends with a newline.
use anyhow::{Context, Result, bail, ensure};
use minijinja::value::{Kwargs, Serde, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value, syntax::SyntaxConfig};
use serde_json::{Map, Value as Json, json};
use std::{collections::HashMap, fs, path::Path};
use tokenizers::Tokenizer;

pub struct Chat {
    env: Environment<'static>,
    tok: Tokenizer,
    bos: Json,
    /// Every token that ends a reply: the tokenizer's eos token (`<|im_end|>`), `<|endoftext|>` and config.json's
    /// eos_token_id.
    ends: Vec<u32>,
}

/// A float as Python's repr writes it: the shortest digits that round-trip, fixed from 1e-4 up to 1e16, else an exponent
/// of at least two digits. Rust and serde_json write 1e-5 and 1e16 where Python writes 1e-05 and 1e+16.
fn py_float(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() { "NaN".into() } else if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let fixed = format!("{x}");
        return if fixed.contains('.') { fixed } else { fixed + ".0" };
    }
    format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// JSON like Python's `json.dumps(v, ensure_ascii=False)`, which HF's `tojson` uses: ", " and ": " separators,
/// insertion order, no HTML escaping.
fn py_json(v: &Json, out: &mut String) {
    match v {
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                py_json(x, out);
            }
            out.push('}');
        }
        Json::Number(n) if n.is_f64() => out.push_str(&py_float(n.as_f64().unwrap())),
        other => out.push_str(&serde_json::to_string(other).unwrap()),
    }
}

/// OpenAI clients send a tool call's arguments as a JSON string, the template wants the object (Transformers fails on
/// a string). `developer` is `system`, `reasoning` is `reasoning_content`.
fn normalize(messages: &Json) -> Result<Json> {
    let mut messages = messages.clone();
    for (i, m) in messages.as_array_mut().context("messages must be an array")?.iter_mut().enumerate() {
        let Some(m) = m.as_object_mut() else { bail!("messages[{i}] must be an object") };
        if m.get("role").and_then(Json::as_str) == Some("developer") {
            m.insert("role".into(), json!("system"));
        }
        if !m.contains_key("reasoning_content")
            && let Some(r) = m.get("reasoning").filter(|r| r.is_string()).cloned()
        {
            m.insert("reasoning_content".into(), r);
        }
        for (j, call) in m.get_mut("tool_calls").and_then(Json::as_array_mut).into_iter().flatten().enumerate() {
            let f = if call.get("function").is_some() { &mut call["function"] } else { call };
            let Some(Json::String(s)) = f.get("arguments") else { continue };
            let args: Json = if s.trim().is_empty() { json!({}) } else {
                serde_json::from_str(s).with_context(|| format!("messages[{i}].tool_calls[{j}].function.arguments is not JSON: {s}"))?
            };
            ensure!(args.is_object(), "messages[{i}].tool_calls[{j}].function.arguments must be a JSON object");
            f["arguments"] = args;
        }
    }
    Ok(messages)
}

/// Remove `{% name %}` tags (any whitespace control): HF's `generation` markers do not render.
fn remove_tags(src: &str, names: &[&str]) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("{%") {
        let Some(j) = rest[i..].find("%}") else { break };
        let inner = rest[i + 2..i + j].trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..i]);
        if !names.contains(&inner) {
            out.push_str(&rest[i..i + j + 2]);
        }
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}

/// Qwen's replies in a rendered conversation (character offsets) as Unsloth's train_on_responses_only marks them,
/// which ft-kit trains with: from after each "<|im_start|>assistant\n" to the next "<|im_start|>user\n" (tool
/// results are user turns) or the end, so the empty think block and <|im_end|> are learned too.
fn response_spans(text: &str) -> Vec<(usize, usize)> {
    const REPLY: &str = "<|im_start|>assistant\n";
    const USER: &str = "<|im_start|>user\n";
    let chars = |b: usize| text[..b].chars().count();
    let mut spans = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find(REPLY) {
        let start = at + i + REPLY.len();
        let end = text[start..].find(USER).map_or(text.len(), |j| start + j);
        spans.push((chars(start), chars(end)));
        at = end;
    }
    spans
}

impl Chat {
    pub fn load(dir: &Path) -> Result<Self> {
        let config: Json = serde_json::from_str(&fs::read_to_string(dir.join("config.json")).context("config.json")?)
            .context("config.json")?;
        let src = fs::read_to_string(dir.join("chat_template.jinja")).context("chat_template.jinja")?;
        // as Transformers renders templates (a line with only a tag or comment leaves nothing behind)
        let mut env = Environment::new();
        env.set_syntax(SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        // Transformers' tojson does not escape non-ASCII: ensure_ascii=False is what it does anyway
        env.add_filter("tojson", |v: Value, kw: Kwargs| -> Result<Value, Error> {
            if kw.get::<Option<bool>>("ensure_ascii")? == Some(true) {
                return Err(Error::new(ErrorKind::InvalidOperation, "tojson: ensure_ascii=True is not supported"));
            }
            kw.assert_all_used()?;
            let j = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
            let mut s = String::new();
            py_json(&j, &mut s);
            Ok(Value::from_safe_string(s))
        });
        // Python's str(): Qwen's template writes a tool call's boolean and null arguments as True, False and None
        env.add_filter("string", |v: Value| -> String {
            match v.kind() {
                ValueKind::None => "None".into(),
                ValueKind::Bool => if v.is_true() { "True" } else { "False" }.into(),
                ValueKind::Number if !v.is_integer() => f64::try_from(v).map_or_else(|_| String::new(), py_float),
                _ => v.to_string(),
            }
        });
        env.add_template_owned("chat", remove_tags(&src, &["generation", "endgeneration"]))?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let cfg: Json = serde_json::from_str(&fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
        // Qwen has none (null) and its template does not use it
        let bos = cfg["bos_token"].clone();
        let token_id = |t: &str| tok.token_to_id(t).with_context(|| format!("no token {t} in tokenizer.json"));
        let eos = token_id(cfg["eos_token"].as_str().context("eos_token")?)?;
        let mut ends = vec![eos];
        // what else ends a reply: the pad token <|endoftext|>, which is also config.json's eos_token_id
        if let Some(pad) = cfg["pad_token"].as_str() {
            ends.push(token_id(pad)?);
        }
        let ids = &config.get("text_config").unwrap_or(&config)["eos_token_id"];
        ends.extend(ids.as_u64().into_iter().chain(ids.as_array().into_iter().flatten().filter_map(Json::as_u64)).map(|i| i as u32));
        ends.sort_unstable();
        ends.dedup();
        Ok(Chat { env, tok, bos, ends })
    }

    /// The token that opens a turn of the conversation (`<|im_start|>`), if the vocabulary has it: a
    /// prompt's last one starts the reply's turn, which the next request renders differently (sessions)
    pub fn turn_token(&self) -> Option<u32> {
        self.tok.token_to_id("<|im_start|>")
    }

    /// Whether `tok` ends a reply (`eos` and `<|endoftext|>`).
    pub fn is_eos(&self, tok: u32) -> bool {
        self.ends.contains(&tok)
    }

    /// The prompt for `messages` (OpenAI format) with the generation prompt; `kwargs` are extra template variables
    /// such as enable_thinking.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, kwargs: &Map<String, Json>) -> Result<String> {
        self.render_template(messages, tools, kwargs, true)
    }

    fn render_template(&self, messages: &Json, tools: Option<&Json>, kwargs: &Map<String, Json>, generation_prompt: bool)
                       -> Result<String> {
        let mut ctx = Map::new();
        // Qwen's template takes arguments as objects and fails on OpenAI's JSON strings
        ctx.insert("messages".into(), normalize(messages)?);
        if let Some(t) = tools {
            ctx.insert("tools".into(), t.clone());
        }
        ctx.insert("add_generation_prompt".into(), json!(generation_prompt));
        ctx.insert("bos_token".into(), self.bos.clone());
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        Ok(self.env.get_template("chat")?.render(Value::from(Serde(&ctx)))?)
    }

    /// Token ids of a whole conversation and which of them the assistant wrote: the replies as Unsloth's
    /// train_on_responses_only marks them (`response_spans`; a token is in the loss if one of its characters is in one).
    pub fn encode_train(&self, messages: &Json, kwargs: &Map<String, Json>) -> Result<(Vec<u32>, Vec<u8>)> {
        self.encode_masked(messages, kwargs, false)
    }

    /// The same with only the last reply in the loss (preference training: a prompt and one response to it, the
    /// prompt's earlier assistant turns are context)
    pub fn encode_last_reply(&self, messages: &Json, kwargs: &Map<String, Json>) -> Result<(Vec<u32>, Vec<u8>)> {
        self.encode_masked(messages, kwargs, true)
    }

    fn encode_masked(&self, messages: &Json, kwargs: &Map<String, Json>, last: bool) -> Result<(Vec<u32>, Vec<u8>)> {
        let text = self.render_template(messages, None, kwargs, false)?;
        let mut spans = response_spans(&text);
        if last {
            spans.drain(..spans.len().saturating_sub(1));
        }
        let enc = self.tok.encode_char_offsets(text.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let mask = enc.get_offsets().iter().map(|&(s, e)| spans.iter().any(|&(a, b)| s < b && e > a) as u8).collect();
        Ok((enc.get_ids().to_vec(), mask))
    }

    /// Token ids of a rendered prompt (special tokens are in the text already).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).unwrap_or_default()
    }

    /// `qwenny chatcheck`: render and tokenize the conversations of a reference from `tools/make_tokref.py` and compare
    /// text and token ids with Transformers'. Returns whether every case is identical.
    pub fn check(&self, reference: &Path) -> Result<bool> {
        let r: Json = serde_json::from_str(&fs::read_to_string(reference).with_context(|| reference.display().to_string())?)?;
        let cases = r["cases"].as_array().context("reference has no cases")?;
        let mut same = 0;
        for c in cases {
            let name = c["name"].as_str().unwrap_or("?");
            let kwargs = c["kwargs"].as_object().cloned().unwrap_or_default();
            let text = self.render(&c["messages"], c["tools"].as_array().map(|_| &c["tools"]), &kwargs);
            let want_text = c["text"].as_str().unwrap_or("");
            let want: Vec<u32> = c["ids"].as_array().context("ids")?.iter().filter_map(|i| i.as_u64().map(|i| i as u32)).collect();
            let (ok, detail) = match text.and_then(|t| Ok((self.encode(&t)?, t))) {
                Err(e) => (false, format!("error: {e:#}")),
                Ok((ids, t)) => {
                    let ok = ids == want && t == want_text;
                    let at = t.chars().zip(want_text.chars()).position(|(a, b)| a != b).unwrap_or(t.chars().count().min(want_text.chars().count()));
                    let around = |s: &str| s.chars().skip(at.saturating_sub(20)).take(60).collect::<String>();
                    let detail = if ok { String::new() } else if t != want_text {
                        format!("text differs at char {at}:\n    ours      {:?}\n    reference {:?}", around(&t), around(want_text))
                    } else {
                        let at = ids.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(ids.len().min(want.len()));
                        format!("ids differ at {at}: ours {:?} reference {:?}", &ids[at..(at + 5).min(ids.len())], &want[at..(at + 5).min(want.len())])
                    };
                    (ok, if ok { format!("{} tokens", ids.len()) } else { detail })
                }
            };
            same += ok as usize;
            println!("{:<9} {name}: {detail}", if ok { "identical" } else { "DIFFERENT" });
        }
        println!("{same}/{} cases identical", cases.len());
        Ok(same == cases.len())
    }
}

#[derive(Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
}

const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAM_START: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";

/// Length of the longest suffix of `s` that is a proper prefix of `tag`.
pub(crate) fn partial_tag(s: &str, tag: &str) -> usize {
    (1..tag.len().min(s.len() + 1)).rev().find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..])).unwrap_or(0)
}

/// A tool call as the model wrote it: `arguments` is a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Json,
}

/// The JSON types a tool parameter's schema allows (`type`, or the `type`s of `anyOf`/`oneOf`); empty: unknown.
fn schema_types(schema: &Json) -> Vec<&str> {
    let mut types: Vec<&str> = match &schema["type"] {
        Json::String(t) => vec![t.as_str()],
        Json::Array(a) => a.iter().filter_map(Json::as_str).collect(),
        _ => vec![],
    };
    for alt in ["anyOf", "oneOf"] {
        for s in schema[alt].as_array().into_iter().flatten() {
            types.extend(schema_types(s));
        }
    }
    types
}

/// A parameter's text as the argument value its schema type asks for: a string stays text, the other types are JSON
/// (booleans and null also as Python wrote them into the prompt: True, False, None); text that is not valid for the
/// type stays a string. No schema: JSON when it parses, else text.
fn param_value(raw: &str, schema: Option<&Json>) -> Json {
    let types = schema.map(schema_types).unwrap_or_default();
    let text = || Json::String(raw.to_string());
    if types.is_empty() {
        return serde_json::from_str(raw).unwrap_or_else(|_| text());
    }
    let t = raw.trim();
    let allows = |k: &str| types.contains(&k);
    let value = match t {
        "None" => Some(Json::Null),
        "True" => Some(json!(true)),
        "False" => Some(json!(false)),
        _ => serde_json::from_str::<Json>(t).ok(),
    };
    let fits = |v: &Json| match v {
        Json::Number(n) => allows("number") || (allows("integer") && (n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0))),
        Json::Array(_) => allows("array"),
        Json::Object(_) => allows("object"),
        Json::Bool(_) => allows("boolean"),
        Json::Null => allows("null"),
        Json::String(_) => allows("string"),
    };
    // a type that also allows strings keeps "123" as text; only null and booleans win over it
    value.filter(|v| fits(v) && (!allows("string") || v.is_null() || v.is_boolean())).unwrap_or_else(text)
}

/// The tools' parameter schemas by function name.
type Schemas = HashMap<String, Json>;

/// `Schemas` from OpenAI's `{"type": "function", "function": {...}}` tools (or the bare function objects).
fn schemas(tools: &Json) -> Schemas {
    let mut map = HashMap::new();
    for t in tools.as_array().into_iter().flatten() {
        let f = t.get("function").unwrap_or(t);
        if let Some(name) = f["name"].as_str() {
            map.insert(name.to_string(), f["parameters"]["properties"].clone());
        }
    }
    map
}

/// `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>` blocks, with the newlines
/// Qwen's template writes around tags and values (one each side of a value belongs to the format). `schemas` type the
/// values (`param_value`). A call needs its `</function>`: a block without it (cut off by the token limit) is dropped.
fn parse_tool_calls(text: &str, schemas: Option<&Schemas>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for block in text.split(TOOL_START).skip(1) {
        let block = block.split(TOOL_END).next().unwrap_or("");
        let Some(f) = block.find(FUNCTION_START) else { continue };
        let rest = &block[f + FUNCTION_START.len()..];
        let Some(end) = rest.find('>') else { continue };
        let body_end = match rest.find(FUNCTION_END) {
            Some(b) if b >= end => b,
            _ => continue,
        };
        let name = rest[..end].trim().to_string();
        let props = schemas.and_then(|s| s.get(&name));
        let mut args = Map::new();
        for p in rest[end + 1..body_end].split(PARAM_START).skip(1) {
            let Some(e) = p.find('>') else { continue };
            let key = p[..e].trim().to_string();
            let raw = p[e + 1..].split(PARAM_END).next().unwrap_or("");
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            args.insert(key.clone(), param_value(raw, props.map(|p| &p[&key]).filter(|s| !s.is_null())));
        }
        calls.push(ToolCall { name, arguments: Json::Object(args) });
    }
    calls
}

/// Splits streamed output into reasoning (before `</think>`, when the prompt opened a think block), content and
/// tool calls; `push` each piece of text, `finish` at the end. Whitespace around the parts (the newlines of the format) is dropped; text that may be the start of a tag, and
/// whitespace that may precede one, waits for the next piece; everything from `<tool_call>` on is held back for
/// `finish` (only when the request has tools: else it is content).
///
/// With thinking off, the base model sometimes thinks anyway and writes its own `</think>`: a stream has sent the
/// text before it as content by then, so the tag stays in the content (the chat page moves it into its thinking box)
/// and `split_stray_think` separates a finished reply.
pub struct OutputParser {
    mode: Mode,
    /// skip whitespace until the next text (after `</think>`, at the start)
    lead: bool,
    pending: String,
    /// the whitespace between the content and `<tool_call>`
    gap: String,
    tool_text: String,
    /// the request's tools' parameter types; `Some` only when the request has tools
    schemas: Option<Schemas>,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Reasoning,
    Content,
    Tools,
}

impl OutputParser {
    /// `thinking`: the prompt ended inside a think block. `tools`: the request's tools (OpenAI format), if any.
    pub fn new(thinking: bool, tools: Option<&Json>) -> Self {
        OutputParser {
            mode: if thinking { Mode::Reasoning } else { Mode::Content },
            lead: true,
            pending: String::new(),
            gap: String::new(),
            tool_text: String::new(),
            schemas: tools.map(schemas),
        }
    }

    pub fn push(&mut self, delta: &str, out: &mut Vec<Piece>) {
        if self.mode == Mode::Tools {
            self.tool_text.push_str(delta);
            return;
        }
        self.pending.push_str(delta);
        loop {
            if self.lead {
                self.pending = self.pending.trim_start().to_string();
                if self.pending.is_empty() {
                    return;
                }
                self.lead = false;
            }
            let tag = match self.mode {
                Mode::Reasoning => Some(THINK_END),
                _ if self.schemas.is_some() => Some(TOOL_START),
                _ => None,
            };
            if let Some(i) = tag.and_then(|t| self.pending.find(t)) {
                let tag = tag.unwrap();
                let after = self.pending[i + tag.len()..].to_string();
                let before = self.pending[..i].trim_end().to_string();
                self.gap = self.pending[before.len()..i].to_string();
                self.pending.clear();
                self.emit(before, out);
                if self.mode == Mode::Reasoning {
                    self.mode = Mode::Content;
                    self.lead = true;
                    self.pending = after;
                    continue;
                }
                self.mode = Mode::Tools;
                self.tool_text = format!("{TOOL_START}{after}");
                return;
            }
            let cut = self.pending.len() - tag.map_or(0, |t| partial_tag(&self.pending, t));
            let ready = self.pending[..cut].trim_end().len();
            let rest = self.pending.split_off(ready);
            let ready = std::mem::replace(&mut self.pending, rest);
            self.emit(ready, out);
            return;
        }
    }

    fn emit(&self, s: String, out: &mut Vec<Piece>) {
        if !s.is_empty() {
            out.push(if self.mode == Mode::Reasoning { Piece::Reasoning(s) } else { Piece::Content(s) });
        }
    }

    /// The reply is complete: flush what was held back and return the tool calls. Held-back text that holds no tool
    /// call (a model talking about the format, a call cut off by the token limit) is content.
    pub fn finish(&mut self, out: &mut Vec<Piece>) -> Vec<ToolCall> {
        let rest = std::mem::take(&mut self.pending);
        self.emit(rest.trim_end().to_string(), out);
        if self.mode != Mode::Tools {
            return Vec::new();
        }
        let calls = parse_tool_calls(&self.tool_text, self.schemas.as_ref());
        if calls.is_empty() {
            self.mode = Mode::Content;
            let held = format!("{}{}", self.gap, std::mem::take(&mut self.tool_text));
            self.emit(held.trim_end().to_string(), out);
        }
        calls
    }
}

/// A finished reply written with thinking off that has a stray `</think>`: (reasoning, content), the text before the
/// tag moved into the reasoning.
pub fn split_stray_think(content: &str) -> (String, String) {
    match content.split_once(THINK_END) {
        Some((r, c)) => (r.trim().to_string(), c.trim().to_string()),
        None => (String::new(), content.to_string()),
    }
}

/// The models the tests run against, from the Hugging Face cache (no download; a test without its model returns early).
#[cfg(test)]
pub mod test_models {
    pub const QWEN: &str = "Qwen/Qwen3.5-4B";

    pub fn dir(id: &str) -> Option<std::path::PathBuf> {
        let d = crate::hub::cached(id).filter(|d| d.join("chat_template.jinja").exists());
        if d.is_none() {
            eprintln!("SKIPPED: {id} is not in the Hugging Face cache, this test checks nothing");
        }
        d
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn floats_as_python_writes_them() {
        for (x, want) in [(1.0, "1.0"), (0.5, "0.5"), (1e-5, "1e-05"), (1.5e-7, "1.5e-07"), (0.0001, "0.0001"), (1e15, "1000000000000000.0"),
                          (1e16, "1e+16"), (1.5e300, "1.5e+300"), (-2.5e-10, "-2.5e-10"), (0.0, "0.0")] {
            assert_eq!(super::py_float(x), want);
        }
    }

    use super::*;

    /// Feed `text` in pieces of `n` characters; the reasoning, the content and the calls.
    fn run(text: &str, thinking: bool, tools: Option<&Json>, n: usize) -> (String, String, Vec<ToolCall>) {
        let mut p = OutputParser::new(thinking, tools);
        let mut pieces = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for c in chars.chunks(n) {
            p.push(&c.iter().collect::<String>(), &mut pieces);
        }
        let calls = p.finish(&mut pieces);
        let (mut r, mut c) = (String::new(), String::new());
        for piece in pieces {
            match piece {
                Piece::Reasoning(s) => r += &s,
                Piece::Content(s) => c += &s,
            }
        }
        (r, c, calls)
    }

    /// The same result however the text is cut into pieces.
    fn all_cuts(text: &str, thinking: bool, tools: Option<&Json>) -> (String, String, Vec<ToolCall>) {
        let whole = run(text, thinking, tools, usize::MAX);
        for n in [1, 2, 3, 5, 8, 13] {
            assert_eq!(run(text, thinking, tools, n), whole, "pieces of {n} characters");
        }
        whole
    }

    fn tools() -> Json {
        json!([
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"},
                "mode": {"type": "integer"}, "ratio": {"type": "number"}, "tags": {"type": "array"},
                "meta": {"type": "object"}, "note": {"type": ["string", "null"]}, "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}},
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
        ])
    }

    #[test]
    fn thinking_and_answer() {
        let (r, c, calls) = all_cuts("Let me think.\nStep 2.\n</think>\n\nThe answer is 4.\n", true, None);
        assert_eq!((r.as_str(), c.as_str()), ("Let me think.\nStep 2.", "The answer is 4."));
        assert!(calls.is_empty());
    }

    #[test]
    fn answer_without_thinking_and_unfinished_thinking() {
        let (r, c, _) = all_cuts("Hello!\n\nSecond paragraph.\n", false, None);
        assert_eq!((r.as_str(), c.as_str()), ("", "Hello!\n\nSecond paragraph."));
        let (r, c, _) = all_cuts("Still thinking about", true, None);
        assert_eq!((r.as_str(), c.as_str()), ("Still thinking about", ""));
    }

    #[test]
    fn stray_think_end_with_thinking_off_stays_in_the_content_for_the_page() {
        let (r, c, _) = all_cuts("Hmm, let me see.\n</think>\n\nIt is 4.", false, None);
        assert_eq!((r.as_str(), c.as_str()), ("", "Hmm, let me see.\n</think>\n\nIt is 4."));
        assert_eq!(split_stray_think(&c), ("Hmm, let me see.".to_string(), "It is 4.".to_string()));
        assert_eq!(split_stray_think("plain"), (String::new(), "plain".to_string()));
    }

    #[test]
    fn tool_call_with_typed_and_multi_line_parameters() {
        let t = tools();
        let text = "Reasoning.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/n.txt\n</parameter>\n\
                    <parameter=content>\nline 1\n  indented\n\nline 4\n\n</parameter>\n<parameter=overwrite>\nTrue\n</parameter>\n\
                    <parameter=mode>\n420\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n<parameter=tags>\n[\"a\", \"b ä\"]\n</parameter>\n\
                    <parameter=meta>\n{\"k\": [1, null]}\n</parameter>\n<parameter=note>\nnull\n</parameter>\n<parameter=limit>\n7\n</parameter>\n\
                    <parameter=unknown>\n12\n</parameter>\n</function>\n</tool_call>";
        let (r, c, calls) = all_cuts(text, true, Some(&t));
        assert_eq!((r.as_str(), c.as_str()), ("Reasoning.", ""));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(
            calls[0].arguments,
            json!({"path": "/tmp/n.txt", "content": "line 1\n  indented\n\nline 4\n", "overwrite": true, "mode": 420, "ratio": 1.5,
                   "tags": ["a", "b ä"], "meta": {"k": [1, null]}, "note": null, "limit": 7, "unknown": 12})
        );
        // key order of the model's text is kept
        assert_eq!(calls[0].arguments.to_string().find("path"), Some(2));
    }

    #[test]
    fn string_parameters_stay_strings_and_bad_values_fall_back_to_text() {
        let t = tools();
        let text = "<tool_call>\n<function=write_file>\n<parameter=path>\n123\n</parameter>\n<parameter=note>\ntrue\n</parameter>\n\
                    <parameter=mode>\nmany\n</parameter>\n<parameter=tags>\nnot a list\n</parameter>\n<parameter=limit>\n2.5\n</parameter>\n</function>\n</tool_call>";
        let (_, _, calls) = all_cuts(text, false, Some(&t));
        assert_eq!(calls[0].arguments, json!({"path": "123", "note": "true", "mode": "many", "tags": "not a list", "limit": "2.5"}));
    }

    #[test]
    fn text_before_a_tool_call() {
        let t = tools();
        let text = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let (r, c, calls) = all_cuts(text, false, Some(&t));
        assert_eq!((r.as_str(), c.as_str()), ("", "Let me check."));
        assert_eq!(calls, vec![ToolCall { name: "get_weather".into(), arguments: json!({"city": "Paris"}) }]);
    }

    #[test]
    fn two_tool_calls() {
        let t = tools();
        let text = "Thinking...\n</think>\n\nBoth cities.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n\
                    <tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>";
        let (r, c, calls) = all_cuts(text, true, Some(&t));
        assert_eq!((r.as_str(), c.as_str()), ("Thinking...", "Both cities."));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
        assert_eq!(calls[1].arguments, json!({"city": "Rome"}));
    }

    #[test]
    fn tool_call_without_parameters_and_cut_off_calls() {
        let t = tools();
        let (_, _, calls) = all_cuts("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>", false, Some(&t));
        assert_eq!(calls, vec![ToolCall { name: "get_weather".into(), arguments: json!({}) }]);
        // cut off by the token limit: no call, the text is content
        let cut = "Hi.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
        let (_, c, calls) = all_cuts(cut, false, Some(&t));
        assert!(calls.is_empty());
        assert_eq!(c, "Hi.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar");
    }

    #[test]
    fn tool_tags_are_content_when_the_request_has_no_tools() {
        let text = "Use <tool_call> blocks like this: <tool_call>x</tool_call>.";
        let (_, c, calls) = all_cuts(text, false, None);
        assert_eq!(c, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn arguments_strings_in_messages_become_objects() {
        let m = json!([{"role": "developer", "content": "x"},
                       {"role": "assistant", "reasoning": "r", "tool_calls": [{"id": "a", "type": "function",
                        "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
                       {"role": "assistant", "tool_calls": [{"function": {"name": "g", "arguments": ""}}]}]);
        let n = normalize(&m).unwrap();
        assert_eq!(n[0]["role"], "system");
        assert_eq!(n[1]["reasoning_content"], "r");
        assert_eq!(n[1]["tool_calls"][0]["function"]["arguments"], json!({"a": 1}));
        assert_eq!(n[2]["tool_calls"][0]["function"]["arguments"], json!({}));
        assert!(normalize(&json!([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}])).is_err());
    }

    /// A call needs its `</function>`; without schemas the values are JSON when they parse, with them typed.
    #[test]
    fn tool_call_parsing_needs_the_closing_tag() {
        let cut = "<tool_call><function=f><parameter=a>\n1\n</parameter><parameter=b>\nx\n</parameter>";
        assert!(parse_tool_calls(cut, None).is_empty());
        let t = schemas(&tools());
        let call = "<tool_call><function=get_weather><parameter=city>\n123\n</parameter></function></tool_call>";
        assert_eq!(parse_tool_calls(call, None)[0].arguments, json!({"city": 123}));
        assert_eq!(parse_tool_calls(call, Some(&t))[0].arguments, json!({"city": "123"}));
    }

    /// Against Transformers' prompts and token ids, when the model and `tools/tokref-qwen.json` are there.
    #[test]
    fn template_and_tokenizer_match_transformers() {
        let reference = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/tokref-qwen.json");
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let chat = Chat::load(&dir).unwrap();
        // <|im_end|> and <|endoftext|> end a reply, <|im_start|> does not
        assert!(chat.is_eos(248046) && chat.is_eos(248044) && !chat.is_eos(248045));
        assert!(chat.check(&reference).unwrap());
    }
}
