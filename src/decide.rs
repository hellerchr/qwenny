//! Decisions by scoring instead of generating (`POST /v1/decisions`): the prompt lists the options under labels
//! (letters, level numbers, or Ja / Nein), the engine prefills it like a chat request and reads the logits at its last
//! position, and a softmax over the labels' tokens is the decision distribution.
use crate::chat::Chat;
use crate::sample::Params;
use crate::server::{App, Job, Msg, bad_request};
use anyhow::{Context, Result, ensure};
use axum::{Json as AxJson, extract::State, http::StatusCode, response::{IntoResponse, Response}};
use serde_json::{Value as Json, json};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// A decision prompt: its tokens and the first token of each option's label
pub struct Item {
    pub ids: Vec<u32>,
    pub cands: Vec<u32>,
}

/// Questions that ask nothing of their own (the page's placeholder)
const GENERIC: [&str; 3] = ["", "Welche Option passt?", "Which option fits?"];

/// The answer words of a yes/no question ("noul"); its two options are what yes and no mean, or these words themselves
const YES_NO: [&str; 2] = ["Ja", "Nein"];

/// The labels the answer is read from, by question type: yes/no reads the words, score the level numbers (0-9 for
/// ten levels), choice the letters.
fn labels(kind: &str, n: usize) -> Result<Vec<String>> {
    Ok(match kind {
        "noul" => YES_NO.iter().map(|w| w.to_string()).collect(),
        "score" if n <= 9 => (1..=n).map(|k| k.to_string()).collect(),
        "score" => (0..n).map(|k| k.to_string()).collect(),
        "choice" => LETTERS.chars().take(n).map(String::from).collect(),
        _ => anyhow::bail!("unknown question type {kind:?} (choice, score or noul)"),
    })
}

/// The decision prompt. The last user message gets the question in front of it and, after it, the question again with
/// the options and the answer instruction; the answer is the first token the assistant would write. The caller's
/// system prompt stays. `decide` and serving build every prompt here, so both see the same text.
/// `type` (default choice): choice options are lettered; score levels are numbered; a yes/no question is answered with
/// the words Ja / Nein, its options say what they mean.
/// "thought" (think, then decide): `true` renders the prompt with thinking on, open for the model to think (the first
/// step); a string is that thinking, closed, and the answer position follows it (the second step).
pub fn build(chat: &Chat, row: &Json) -> Result<Item> {
    let options: Vec<&str> = row["options"].as_array().context("options")?.iter().filter_map(|o| o.as_str()).collect();
    let kind = row["type"].as_str().unwrap_or("choice");
    ensure!(options.len() >= 2 && options.len() <= LETTERS.len(), "{} options", options.len());
    ensure!(kind != "noul" || options.len() == 2, "a yes/no question has two options");
    ensure!(kind != "score" || options.len() <= 10, "a score question has 2 to 10 levels");
    let labels = labels(kind, options.len())?;
    let mut question = row["question"].as_str().unwrap_or("").trim().to_string();
    let mut msgs = row["messages"].as_array().context("messages")?.clone();
    let last = msgs.iter_mut().rev().find(|m| m["role"] == "user").context("no user message")?;
    let mut content = last["content"].as_str().unwrap_or("").trim().to_string();
    // no question of its own (empty, or the page's placeholder): the content is the question
    if GENERIC.contains(&question.as_str()) {
        question = if content.is_empty() { GENERIC[1].to_string() } else { std::mem::take(&mut content) };
    }
    let question = question.as_str();
    // the question first, so the model reads the content knowing what is asked (not when the content is the question)
    let mut text = if content.is_empty() || content == question.trim() { String::new() } else { format!("Frage: {question}\n\n{content}\n\n") };
    text.push_str(&format!("Frage: {question}\n"));
    match kind {
        "noul" if options != YES_NO => text.push_str(&format!("Ja bedeutet: {}\nNein bedeutet: {}\n", options[0], options[1])),
        "noul" => {}
        _ => {
            for (o, l) in options.iter().zip(&labels) {
                text.push_str(&format!("{l} = {o}\n"));
            }
        }
    }
    text.push_str(match kind {
        "noul" => "Antworte nur mit Ja oder Nein.",
        "score" => "Antworte nur mit der Zahl der passenden Stufe.",
        _ => "Antworte nur mit dem Buchstaben der passenden Option.",
    });
    last["content"] = json!(text);
    let thought = row.get("thought").filter(|t| t.is_string() || t == &&Json::Bool(true));
    let kwargs = json!({"enable_thinking": thought.is_some()}).as_object().unwrap().clone();
    let mut prompt = chat.render(&Json::Array(msgs), None, &kwargs)?;
    if let Some(t) = thought.and_then(Json::as_str) {
        prompt.push_str(t.trim_end());
        prompt.push_str("</think>");
    }
    let ids = chat.encode(&prompt)?;
    let mut cands = Vec::new();
    for l in &labels {
        // a label's first token after the prompt (Nein may be more than one)
        let with = chat.encode(&format!("{prompt}{l}"))?;
        ensure!(with.len() > ids.len() && with[..ids.len()] == ids[..], "label {l} does not start a new token after the prompt");
        ensure!(!cands.contains(&with[ids.len()]), "labels {labels:?} share a first token");
        cands.push(with[ids.len()]);
    }
    Ok(Item { ids, cands })
}

/// `POST /v1/decisions`: {"messages", "question", "options"} (or "preset": a name from --decisions filling in what is
/// missing), optional "type", "temperature" (calibration, default 1), "threshold" (lowest top probability to act on)
/// and "min_margin" (lowest gap to the runner-up). The types follow Jev: "choice" (default) picks one option, "score"
/// places the input on ordered levels (low to high, 2 to 10) and adds "score" (the probability-weighted level, 0 to
/// 1), "noul" is a yes/no question and adds "noul" (the probability of yes). Options are strings, {"key",
/// "description"} objects or a map key -> description ("criteria"; for "noul" {"true", "false"}). Returns the choice,
/// every option's probability, the margin, a confidence (1 - normalized entropy), the label mass and, with a threshold,
/// "decision": "auto" or "escalate". With "questions" ({name: {"type", "question", "options", ...}}) every question
/// is asked about the same messages in one call (scored side by side) and the reply has "answers" by name.
pub async fn decisions(State(app): State<Arc<App>>, AxJson(body): AxJson<Json>) -> Response {
    let t0 = std::time::Instant::now();
    let mut req = body.clone();
    if let Some(name) = body.get("preset").and_then(Json::as_str) {
        let Some(p) = app.presets.as_array().and_then(|a| a.iter().find(|p| p["name"] == name)) else {
            return bad_request(format!("no preset {name:?}"));
        };
        for (k, v) in p.as_object().unwrap() {
            if req.get(k).is_none_or(Json::is_null) {
                req[k] = v.clone();
            }
        }
        // a preset's system prompt goes first unless the request brings one
        if let Some(sys) = p.get("system").and_then(Json::as_str)
            && let Some(msgs) = req["messages"].as_array_mut().filter(|m| m.first().is_none_or(|m| m["role"] != "system"))
        {
            msgs.insert(0, json!({"role": "system", "content": sys}));
        }
    }
    let many = req.get("questions").and_then(Json::as_object).cloned();
    let mut rows = Vec::new();
    match &many {
        Some(qs) => {
            for (name, q) in qs {
                let mut row = req.clone();
                row.as_object_mut().unwrap().remove("questions");
                for (k, v) in q.as_object().into_iter().flatten() {
                    row[k] = v.clone();
                }
                rows.push((name.clone(), row));
            }
        }
        None => rows.push((String::new(), req.clone())),
    }
    if rows.is_empty() {
        return bad_request("no questions".into());
    }
    let model = match app.model(&req) {
        Ok(m) => m.1,
        Err(e) => return bad_request(format!("{e:#}")),
    };
    // "think": true (or a token budget): each such question is thought about first, all at once
    let mut thoughts = std::collections::HashMap::new();
    let mut thinking = Vec::new();
    for (i, (_, row)) in rows.iter().enumerate() {
        let Some(budget) = think_budget(row) else { continue };
        let (app, mut row) = (app.clone(), row.clone());
        thinking.push((i, tokio::spawn(async move { think_first(&app, &mut row, budget).await.map(|t| (row, t)) })));
    }
    for (i, task) in thinking {
        match task.await {
            Ok(Ok((row, t))) => {
                rows[i].1 = row;
                thoughts.insert(rows[i].0.clone(), t);
            }
            Ok(Err(e)) => return bad_request(format!("thinking: {e}")),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }
    // every question goes to the engine before any answer is read: they are prefilled side by side
    let mut asked = Vec::new();
    for (name, row) in rows {
        match ask(&app, row) {
            Ok((a, rx)) => asked.push((name, a, rx)),
            Err(e) => return if many.is_some() { bad_request(format!("question {name:?}: {e}")) } else { bad_request(e) },
        }
    }
    let mut answers = serde_json::Map::new();
    let mut tokens = 0;
    for (name, a, rx) in asked {
        let (logits, label_mass, prompt_tokens) = match decision_reply(rx).await {
            Ok(d) => d,
            Err(e) => return e,
        };
        tokens += prompt_tokens;
        let mut ans = answer(&a, &logits, label_mass, prompt_tokens);
        if let Some(t) = thoughts.remove(&name) {
            ans["thinking"] = json!(t);
        }
        answers.insert(name, ans);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut out = match many {
        Some(_) => json!({"object": "decisions", "model": model, "answers": answers,
                          "usage": {"prompt_tokens": tokens, "completion_tokens": 0}}),
        None => answers.into_iter().next().unwrap().1,
    };
    out["model"] = json!(model);
    out["ms"] = json!(ms);
    if let Some(id) = body.get("id") {
        out["id"] = id.clone();
    }
    AxJson(out).into_response()
}

/// "think": true (2,048 tokens) or a token budget
fn think_budget(row: &Json) -> Option<usize> {
    match row.get("think") {
        Some(Json::Bool(true)) => Some(2048),
        Some(Json::Number(n)) => n.as_u64().filter(|&n| n > 0).map(|n| n as usize),
        _ => None,
    }
}

/// One question of a decision request, sent to the engine.
struct Asked {
    kind: String,
    keys: Vec<String>,
    row: Json,
}

async fn decision_reply(mut rx: UnboundedReceiver<Msg>) -> Result<(Vec<f32>, f64, usize), Response> {
    match rx.recv().await {
        Some(Msg::Decision { logits, label_mass, prompt_tokens }) => Ok((logits, label_mass, prompt_tokens)),
        Some(Msg::Error(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, e).into_response()),
        _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "engine dropped the request").into_response()),
    }
}

/// The option keys and the texts the model reads, from any of the forms `decisions` accepts.
fn decision_options(kind: &str, row: &Json) -> Result<(Vec<String>, Vec<String>), String> {
    let text = |key: &str, d: Option<&Json>| d.and_then(Json::as_str).filter(|d| !d.trim().is_empty()).unwrap_or(key).to_string();
    let opts = row.get("options").filter(|o| !o.is_null()).or(row.get("criteria")).cloned().unwrap_or(Json::Null);
    let pairs: Vec<(String, String)> = match (&opts, kind) {
        (Json::Null, "noul") => vec![("yes".into(), "Ja".into()), ("no".into(), "Nein".into())],
        (Json::Object(m), "noul") => vec![("yes".into(), text("Ja", m.get("true"))), ("no".into(), text("Nein", m.get("false")))],
        (Json::Object(m), _) => m.iter().map(|(k, v)| (k.clone(), text(k, Some(v)))).collect(),
        (Json::Array(a), _) => a.iter().map(|o| match o {
            Json::String(s) => Ok((s.clone(), s.clone())),
            Json::Object(m) => {
                let key = m.get("key").and_then(Json::as_str).ok_or("an option object without \"key\"")?;
                Ok((key.to_string(), text(key, m.get("description"))))
            }
            _ => Err("an option is neither a string nor {\"key\", \"description\"}"),
        }).collect::<Result<_, _>>()?,
        _ => return Err("no \"options\"".into()),
    };
    match kind {
        "choice" => {}
        "score" if (2..=10).contains(&pairs.len()) => {}
        "score" => return Err(format!("a score question has 2 to 10 levels, not {}", pairs.len())),
        "noul" if pairs.len() == 2 => {}
        "noul" => return Err("a yes/no question has two options".into()),
        _ => return Err(format!("unknown question type {kind:?} (choice, score or noul)")),
    }
    Ok(pairs.into_iter().unzip())
}

/// Build one question's prompt and send it to the engine.
fn ask(app: &App, mut row: Json) -> Result<(Asked, UnboundedReceiver<Msg>), String> {
    if row.get("question").is_none_or(Json::is_null)
        && let Some(i) = row.get("instructions").cloned()
    {
        row["question"] = i;
    }
    let kind = row.get("type").and_then(Json::as_str).unwrap_or("choice").to_string();
    let (keys, texts) = decision_options(&kind, &row)?;
    row["options"] = json!(texts);
    let item = build(&app.chat, &row).map_err(|e| format!("{e:#}"))?;
    let adapter = app.model(&row).map_err(|e| format!("{e:#}"))?.0;
    send_decision(app, item, Asked { kind, keys, row }, adapter)
}

/// Send a built decision to the engine.
fn send_decision(app: &App, item: Item, asked: Asked, adapter: bool) -> Result<(Asked, UnboundedReceiver<Msg>), String> {
    if item.ids.len() + 1 >= app.max_ctx {
        return Err(format!("prompt of {} tokens does not fit the context of {}", item.ids.len(), app.max_ctx));
    }
    let (tx, rx) = unbounded_channel();
    let params = Params { temperature: 0.0, top_p: 1.0, top_k: 1, repetition_penalty: 1.0, presence_penalty: 0.0, seed: 1 };
    let job = Job { prompt: item.ids, params, max_tokens: 1, stop: vec![], progress: None, ignore_eos: false, decide: Some(item.cands), adapter,
                    tx };
    app.jobs.send(job).map_err(|_| "engine stopped".to_string())?;
    Ok((asked, rx))
}

/// Think, then decide: the model thinks about the decision prompt for at most
/// `budget` tokens, then the options are scored at the answer position after that thinking. Returns the thinking.
async fn think_first(app: &App, row: &mut Json, budget: usize) -> Result<String, String> {
    let kind = row.get("type").and_then(Json::as_str).unwrap_or("choice").to_string();
    let (_, texts) = decision_options(&kind, row)?;
    let mut open = row.clone();
    open["options"] = json!(texts);
    open["thought"] = json!(true);
    let item = build(&app.chat, &open).map_err(|e| format!("{e:#}"))?;
    let adapter = app.model(row).map_err(|e| format!("{e:#}"))?.0;
    let thought = think(app, item.ids, budget, adapter).await?;
    row["thought"] = json!(thought);
    Ok(thought)
}

/// The model's thinking after a prompt that opens a think block, at most `budget` tokens (greedy, up to
/// `</think>`; cut off, it is told to decide now)
async fn think(app: &App, prompt: Vec<u32>, budget: usize, adapter: bool) -> Result<String, String> {
    let (tx, mut rx) = unbounded_channel();
    let params = Params { temperature: 0.0, top_p: 1.0, top_k: 1, repetition_penalty: 1.0, presence_penalty: 0.0, seed: 1 };
    let max_tokens = budget.min(app.max_ctx.saturating_sub(prompt.len() + 64)).max(1);
    let job = Job { prompt, params, max_tokens, stop: vec!["</think>".into()], progress: None, ignore_eos: false, decide: None, adapter, tx };
    app.jobs.send(job).map_err(|_| "engine stopped".to_string())?;
    let mut thought = String::new();
    while let Some(m) = rx.recv().await {
        match m {
            Msg::Text(t) => thought.push_str(&t),
            // out of budget mid-thought: close it, so the answer does not just follow the last word
            Msg::Done { finish: "length", .. } => {
                thought.push_str("\n\nI have to decide now.");
                break;
            }
            Msg::Done { .. } => break,
            Msg::Error(e) => return Err(e),
            Msg::Decision { .. } => {}
        }
    }
    Ok(thought)
}

/// The reply for one question from its candidate logits.
fn answer(a: &Asked, logits: &[f32], label_mass: f64, prompt_tokens: usize) -> Json {
    let t = a.row.get("temperature").and_then(Json::as_f64).unwrap_or(1.0).max(1e-3);
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = logits.iter().map(|&x| ((x as f64 - mx) / t).exp()).collect();
    let sum: f64 = e.iter().sum();
    let p: Vec<f64> = e.iter().map(|x| x / sum).collect();
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let (top, margin) = (order[0], p[order[0]] - p.get(order[1]).copied().unwrap_or(0.0));
    let entropy: f64 = p.iter().filter(|&&q| q > 0.0).map(|&q| -q * q.ln()).sum();
    let mut out = json!({
        "object": "decision", "type": a.kind, "choice": a.keys[top], "index": top,
        "probs": a.keys.iter().zip(&p).map(|(o, q)| json!({"option": o, "p": q})).collect::<Vec<_>>(),
        "margin": margin, "confidence": 1.0 - entropy / (p.len() as f64).ln(), "label_mass": label_mass, "temperature": t,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
    });
    match a.kind.as_str() {
        "score" => out["score"] = json!(p.iter().enumerate().map(|(k, q)| q * k as f64).sum::<f64>() / (p.len() - 1) as f64),
        "noul" => out["noul"] = json!(p[0]),
        _ => {}
    }
    // on a scale of 4 or more levels a neighbouring level is a small error: the threshold is compared with the most
    // likely level and its neighbours ("p_near", sure up to one level), and the margin to the runner-up is not asked
    let near = (a.kind == "score" && p.len() >= 4).then(|| p[top.saturating_sub(1)..(top + 2).min(p.len())].iter().sum::<f64>());
    if let Some(n) = near {
        out["p_near"] = json!(n);
    }
    if let Some(th) = a.row.get("threshold").and_then(Json::as_f64) {
        let min_margin = a.row.get("min_margin").and_then(Json::as_f64).unwrap_or(0.0);
        let act = match near {
            Some(n) => n >= th,
            None => p[top] >= th && margin >= min_margin,
        };
        out["decision"] = json!(if act { "auto" } else { "escalate" });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::test_models;

    /// A decision prompt works with the chat template: every option letter is one token after the prompt
    /// (`build` checks it), and the prompt ends where the answer starts. Skipped without the model.
    #[test]
    fn decision_prompts_build() {
        let row = json!({"messages": [{"role": "user", "content": "Mein Router ist kaputt."}], "question": "Was tun?",
                         "options": ["neu starten", "Techniker rufen", "ignorieren"]});
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let chat = Chat::load(&dir).unwrap();
        let item = build(&chat, &row).unwrap_or_else(|e| panic!("{}: {e:#}", dir.display()));
        assert_eq!(item.cands.len(), 3);
        let tail = chat.decode(&item.ids[item.ids.len().saturating_sub(12)..]);
        assert!(tail.contains("assistant"), "{tail:?}");
    }
}
