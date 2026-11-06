//! The command line. Each command declares its arguments and options with their help (main.rs's `APP`); parsing, the
//! checks before anything runs, the defaults and `--help` all come from those declarations, and the tests below hold
//! every declaration to being documented.
use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};

/// The width help is wrapped to (rustfmt's, too)
const WIDTH: usize = 120;
/// Where the descriptions of a command list or option list start
const COLUMN: usize = 24;

pub struct App {
    pub name: &'static str,
    /// The first lines of `--help`
    pub about: &'static str,
    /// The last lines of `--help`: what holds for every command
    pub notes: &'static str,
    /// The first lines of `help checks`
    pub checks: &'static str,
    pub commands: &'static [Command],
}

pub type Run = fn(&Opts) -> Result<()>;

pub struct Command {
    pub name: &'static str,
    pub group: Group,
    /// One line: what it does
    pub summary: &'static str,
    /// More for `help <command>` ("" for none); a line break starts a new line
    pub about: &'static str,
    pub args: &'static [Arg],
    /// Its options: groups shared with other commands, then its own
    pub opts: &'static [&'static [Opt]],
    pub run: Run,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    Main,
    Research,
    Check,
}

/// A positional argument
pub struct Arg {
    pub name: &'static str,
    pub optional: bool,
    pub help: &'static str,
}

pub struct Opt {
    pub name: &'static str,
    /// What the value is, as `--name=<value>` shows it; one without `<` lists the choices (`cuda|mlx`); "" for a
    /// switch, which takes no value
    pub value: &'static str,
    /// What `Opts::get` returns when it is not given, "" for nothing (then the help says what happens without it)
    pub default: &'static str,
    pub help: &'static str,
}

pub const fn arg(name: &'static str, help: &'static str) -> Arg {
    Arg { name, optional: false, help }
}

pub const fn optional(name: &'static str, help: &'static str) -> Arg {
    Arg { name, optional: true, help }
}

pub const fn opt(name: &'static str, value: &'static str, default: &'static str, help: &'static str) -> Opt {
    Opt { name, value, default, help }
}

pub const fn switch(name: &'static str, help: &'static str) -> Opt {
    Opt { name, value: "", default: "", help }
}

impl Opt {
    fn is_switch(&self) -> bool {
        self.value.is_empty()
    }

    fn choices(&self) -> Option<std::str::Split<'static, char>> {
        (!self.is_switch() && !self.value.contains('<')).then(|| self.value.split('|'))
    }

    fn shown(&self) -> String {
        if self.is_switch() { format!("--{}", self.name) } else { format!("--{}={}", self.name, self.value) }
    }
}

impl Command {
    pub fn opt(&self, name: &str) -> Option<&'static Opt> {
        self.options().find(|o| o.name == name)
    }

    fn options(&self) -> impl Iterator<Item = &'static Opt> {
        self.opts.iter().flat_map(|g| g.iter())
    }

    fn usage(&self, app: &str) -> String {
        let mut u = format!("{app} {}", self.name);
        for a in self.args {
            u += &if a.optional { format!(" [{}]", a.name) } else { format!(" <{}>", a.name) };
        }
        if self.options().next().is_some() {
            u += " [options]";
        }
        u
    }

    pub fn help(&self, app: &str) -> String {
        let mut h = format!("{}\n{}\n", self.usage(app), wrap(self.summary, 0, 0));
        if !self.about.is_empty() {
            h += &format!("\n{}\n", wrap(self.about, 0, 0));
        }
        if !self.args.is_empty() {
            h += "\narguments:\n";
            for a in self.args {
                h += &row(&if a.optional { format!("[{}]", a.name) } else { format!("<{}>", a.name) }, a.help);
            }
        }
        if self.options().next().is_some() {
            h += "\noptions:\n";
            for o in self.options() {
                let default = if o.default.is_empty() { String::new() } else { format!(" (default: {})", o.default) };
                h += &row(&o.shown(), &format!("{}{default}", o.help));
            }
        }
        h.trim_end().to_string()
    }
}

impl App {
    fn command(&self, name: &str) -> Option<&'static Command> {
        self.commands.iter().find(|c| c.name == name)
    }

    fn in_group(&self, g: Group) -> impl Iterator<Item = &'static Command> {
        self.commands.iter().filter(move |c| c.group == g)
    }

    pub fn help(&self) -> String {
        let mut h = format!("{}\n", wrap(self.about, 0, 0));
        for (g, title) in [(Group::Main, "commands"), (Group::Research, "research commands")] {
            h += &format!("\n{title}:\n");
            for c in self.in_group(g) {
                h += &row(c.name, c.summary);
            }
        }
        let checks: Vec<&str> = self.in_group(Group::Check).map(|c| c.name).collect();
        h += &format!("\n{}\n", wrap(&format!("checks ({} help checks): {}", self.name, checks.join(", ")), 0, 2));
        h += &format!("\n{}", wrap(self.notes, 0, 0));
        h
    }

    pub fn checks_help(&self) -> String {
        let mut h = format!("{}\n\n", wrap(self.checks, 0, 0));
        for c in self.in_group(Group::Check) {
            h += &row(c.name, c.summary);
        }
        h.trim_end().to_string()
    }

    fn see_help(&self) -> String {
        format!("{} --help lists the commands", self.name)
    }
}

/// `text` wrapped to WIDTH: its first line starts at column `at`, every further line at `indent`
fn wrap(text: &str, at: usize, indent: usize) -> String {
    let mut out = String::new();
    let mut col = at;
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out += &format!("\n{}", " ".repeat(indent));
            col = indent;
        }
        let mut fresh = true;
        for w in line.split_whitespace() {
            let n = w.chars().count();
            if !fresh && col + 1 + n > WIDTH {
                out += &format!("\n{}", " ".repeat(indent));
                col = indent;
            } else if !fresh {
                out.push(' ');
                col += 1;
            }
            out += w;
            col += n;
            fresh = false;
        }
    }
    out
}

/// One line of a list: `name` and, from COLUMN on (on the next line when `name` reaches it), its description
fn row(name: &str, text: &str) -> String {
    let lead = format!("  {name}");
    let n = lead.chars().count();
    let lead = if n + 2 > COLUMN { format!("{lead}\n{}", " ".repeat(COLUMN)) } else { lead + &" ".repeat(COLUMN - n) };
    format!("{lead}{}\n", wrap(text, COLUMN, COLUMN))
}

/// A command line, checked against its command's declarations
pub struct Opts {
    pub cmd: &'static Command,
    pub args: Vec<String>,
    values: Vec<(String, String)>,
    switches: Vec<String>,
}

impl Opts {
    /// The value given for --key (an option the command does not declare is never given)
    pub fn given(&self, key: &str) -> Option<&str> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn has(&self, key: &str) -> bool {
        self.given(key).is_some()
    }

    pub fn switch(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    pub fn path(&self, key: &str) -> Option<PathBuf> {
        self.given(key).map(PathBuf::from)
    }

    /// The value given for --key, None when it is not (its default does not count)
    pub fn opt<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>> {
        self.given(key).map(|v| v.parse().map_err(|_| anyhow::anyhow!("bad value for --{key}: {v}"))).transpose()
    }

    /// The value of --key: the one given, else its default; an error when it has neither
    pub fn get<T: std::str::FromStr>(&self, key: &str) -> Result<T> {
        let o = self.cmd.opt(key).unwrap_or_else(|| panic!("{} reads --{key} without declaring it", self.cmd.name));
        if let Some(v) = self.opt(key)? {
            return Ok(v);
        }
        ensure!(!o.default.is_empty(), "{} needs {}: {}", self.cmd.name, o.shown(), o.help);
        o.default.parse().map_err(|_| anyhow::anyhow!("--{key}'s default {} is not a value", o.default))
    }

    pub fn arg(&self, i: usize) -> &str {
        &self.args[i]
    }

    pub fn arg_path(&self, i: usize) -> &Path {
        Path::new(&self.args[i])
    }
}

pub enum Parsed {
    Help(String),
    Run(Opts),
}

/// `argv` (without the program) as a command to run, or the help it asks for
pub fn parse(app: &'static App, argv: &[String]) -> Result<Parsed> {
    // a switch takes no value in any command, so `--switch arg` leaves the argument alone before the command is known
    let is_switch = |n: &str| app.commands.iter().any(|c| c.opt(n).is_some_and(Opt::is_switch));
    let (mut args, mut values, mut flags, mut help) = (Vec::new(), Vec::<(String, String)>::new(), Vec::new(), false);
    let mut it = argv.iter().peekable();
    while let Some(a) = it.next() {
        if a == "--help" || a == "-h" {
            help = true;
            continue;
        }
        let Some(f) = a.strip_prefix("--") else {
            args.push(a.clone());
            continue;
        };
        match f.split_once('=') {
            Some((k, v)) => values.push((k.to_string(), tilde(v))),
            None if is_switch(f) => flags.push(f.to_string()),
            None => match it.next_if(|v| !v.starts_with("--")) {
                Some(v) => values.push((f.to_string(), v.clone())),
                None => flags.push(f.to_string()),
            },
        }
    }
    if help || args.first().is_none_or(|a| a == "help") {
        return Ok(Parsed::Help(match args.iter().find(|a| *a != "help").map(|s| s.as_str()) {
            None => app.help(),
            Some("checks") => app.checks_help(),
            Some(t) => app.command(t).with_context(|| format!("no command {t}: {}", app.see_help()))?.help(app.name),
        }));
    }
    let cmd = app.command(&args[0]).with_context(|| format!("no command {}: {}", args[0], app.see_help()))?;
    let name = cmd.name;
    let names = || cmd.options().map(|o| o.name);
    for (i, (k, v)) in values.iter().enumerate() {
        let o = cmd.opt(k).with_context(|| format!("{name} has no option --{k}{}", did_you_mean(k, names())))?;
        ensure!(!o.is_switch(), "--{k} takes no value");
        ensure!(!values[..i].iter().any(|(e, _)| e == k), "--{k} is given twice");
        if let Some(mut choices) = o.choices() {
            ensure!(choices.any(|c| c == v), "--{k} is one of {}, not {v}", o.value.replace('|', ", "));
        }
    }
    for f in &flags {
        let o = cmd.opt(f).with_context(|| format!("{name} has no option --{f}{}", did_you_mean(f, names())))?;
        ensure!(o.is_switch(), "--{f} needs a value: {}", o.shown());
    }
    let args = args.split_off(1);
    let usage = || format!("usage: {} ({} help {name})", cmd.usage(app.name), app.name);
    let required = cmd.args.iter().filter(|a| !a.optional).count();
    ensure!(args.len() >= required, "{name} needs <{}>; {}", cmd.args[args.len()].name, usage());
    if args.len() > cmd.args.len() {
        if let Some(dir) = args.iter().find(|x| Path::new(x).join("config.json").exists()) {
            bail!("{name}: {dir} is a model: give it as --model={dir}");
        }
        bail!("{name}: unexpected arguments {:?}; {}", &args[cmd.args.len()..], usage());
    }
    Ok(Parsed::Run(Opts { cmd, args, values, switches: flags }))
}

/// `~/x` as `$HOME/x` (the shell leaves a ~ alone after `=`)
pub fn tilde(p: &str) -> String {
    match (p.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => p.to_string(),
    }
}

/// " (did you mean --x?)" for the closest name within two edits, else ""
fn did_you_mean<'a>(k: &str, names: impl Iterator<Item = &'a str>) -> String {
    let dist = |a: &str, b: &str| {
        let b: Vec<char> = b.chars().collect();
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut prev = row[0];
            row[0] = i + 1;
            for j in 0..b.len() {
                let cur = row[j + 1];
                row[j + 1] = (prev + (ca != b[j]) as usize).min(row[j] + 1).min(cur + 1);
                prev = cur;
            }
        }
        row[b.len()]
    };
    names
        .map(|n| (dist(k, n), n))
        .filter(|(d, _)| *d <= 2)
        .min()
        .map_or(String::new(), |(_, n)| format!(" (did you mean --{n}?)"))
}

/// The lint of the command line's documentation: every command, argument and option is described, the help fits the
/// terminal, and what is declared is used
#[cfg(test)]
mod tests {
    use super::*;
    use crate::APP;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn every_command_argument_and_option_is_described() {
        let kebab =
            |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        let sentence = |s: &str| !s.trim().is_empty() && !s.ends_with('.') && s.trim() == s;
        for c in APP.commands {
            let n = c.name;
            assert!(kebab(n) && n != "help" && n != "checks", "{n}: a command name is lowercase words with -");
            assert_eq!(APP.commands.iter().filter(|d| d.name == n).count(), 1, "{n} is declared twice");
            assert!(sentence(c.summary) && !c.summary.contains('\n'), "{n}: the summary is one line, no full stop");
            assert!(c.about.trim() == c.about, "{n}: the about text has spaces around it");
            for a in c.args {
                assert!(sentence(a.help), "{n} <{}>: undescribed", a.name);
            }
            let first_optional = c.args.iter().position(|a| a.optional).unwrap_or(c.args.len());
            assert!(c.args[first_optional..].iter().all(|a| a.optional), "{n}: an optional argument before a required");
            for o in c.options() {
                let k = o.name;
                assert!(kebab(k), "{n} --{k}: an option name is lowercase words with -");
                assert_eq!(c.options().filter(|p| p.name == k).count(), 1, "{n} --{k} is declared twice");
                assert!(sentence(o.help), "{n} --{k}: undescribed");
                if o.is_switch() {
                    assert!(o.default.is_empty(), "{n} --{k}: a switch has no default");
                } else if o.default.is_empty() {
                    assert!(
                        ["default", "required", "without it"].iter().any(|w| o.help.contains(w)),
                        "{n} --{k}: no default, and the help does not say what happens without it"
                    );
                }
                if let Some(mut choices) = o.choices().filter(|_| !o.default.is_empty()) {
                    assert!(choices.any(|v| v == o.default), "{n} --{k}: the default is not one of the choices");
                }
            }
        }
    }

    #[test]
    fn a_switch_is_one_in_every_command() {
        let all = || APP.commands.iter().flat_map(|c| c.options());
        for o in all() {
            assert!(all().filter(|p| p.name == o.name).all(|p| p.is_switch() == o.is_switch()), "--{}", o.name);
        }
    }

    #[test]
    fn the_help_fits_and_lists_every_command() {
        let mut pages = vec![("--help".to_string(), APP.help()), ("help checks".into(), APP.checks_help())];
        pages.extend(APP.commands.iter().map(|c| (format!("help {}", c.name), c.help(APP.name))));
        for (page, text) in &pages {
            for line in text.lines() {
                assert!(line.chars().count() <= WIDTH, "{page}: wider than {WIDTH}: {line}");
                assert_eq!(line.trim_end(), line, "{page}: trailing spaces: {line}");
            }
        }
        for c in APP.commands {
            let page = if c.group == Group::Check { &pages[1].1 } else { &pages[0].1 };
            assert!(page.contains(&format!("  {} ", c.name)), "{} is missing from its list", c.name);
            assert!(pages[0].1.contains(c.name), "--help does not name {}", c.name);
        }
    }

    /// Each option is read somewhere: a declaration nothing reads documents what does not exist
    #[test]
    fn every_option_is_read() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let code: String = std::fs::read_dir(&src)
            .unwrap()
            .map(|e| e.unwrap().path())
            // this file's tests read options too
            .filter(|p| p.extension().is_some_and(|e| e == "rs") && !p.ends_with("cli.rs"))
            .map(|p| std::fs::read_to_string(p).unwrap())
            .collect();
        // a declaration is opt("name", ...); a read is get("name"), switch("name") and the like
        for o in APP.commands.iter().flat_map(|c| c.options()) {
            assert!(code.contains(&format!("(\"{}\")", o.name)), "--{} is declared but never read", o.name);
        }
    }

    #[test]
    fn parsing_holds_to_the_declarations() {
        let run = |s: &str| match parse(&APP, &argv(s)) {
            Ok(Parsed::Run(o)) => Ok(o),
            Ok(Parsed::Help(_)) => panic!("{s}: help"),
            Err(e) => Err(e.to_string()),
        };
        let o = run("abliterate a.jsonl b.jsonl out --thinking --alpha 1.5").unwrap();
        assert!(o.switch("thinking") && !o.switch("bf16"));
        assert_eq!((o.get::<f32>("alpha").unwrap(), o.arg(2)), (1.5, "out"));
        assert_eq!(run("bake-direction r.safetensors out").unwrap().get::<f32>("alpha").unwrap(), 1.0);
        assert_eq!(run("serve").unwrap().opt::<usize>("slots").unwrap(), None);
        assert_eq!(run("serve").unwrap().get::<usize>("slots").unwrap(), 16);
        let err = |s: &str| run(s).err().unwrap();
        assert!(err("serve --slot=2").contains("did you mean --slots?"));
        assert!(err("serve --port").contains("needs a value"));
        assert!(err("serve --bf16=1").contains("takes no value"));
        assert!(err("serve --port=1 --port=2").contains("given twice"));
        assert!(err("serve --backend=rocm").contains("one of cuda, mlx"));
        assert!(err("abliterate a.jsonl").contains("needs <harmless.jsonl>"));
        assert!(err("merge a b c").contains("unexpected arguments"));
        assert!(err("nope").contains("no command nope"));
        assert!(run("longcheck").unwrap().get::<f32>("limit").unwrap_err().to_string().contains("needs --limit="));
        for (s, start) in [
            ("", "qwenny:"),
            ("help checks", "checks:"),
            ("serve --help", "qwenny serve"),
            ("help train", "qwenny train"),
        ] {
            match parse(&APP, &argv(s)) {
                Ok(Parsed::Help(h)) => assert!(h.starts_with(start), "{s}: {h}"),
                _ => panic!("{s}: no help"),
            }
        }
    }

    #[test]
    fn wrapping_keeps_the_width_and_the_indent() {
        let w = wrap(&"word ".repeat(60), 10, 4);
        let lines: Vec<&str> = w.lines().collect();
        assert!(lines.len() > 2 && lines[0].len() <= WIDTH - 10);
        assert!(lines[1..].iter().all(|l| l.starts_with("    w") && l.len() <= WIDTH));
        assert!(lines[1..lines.len() - 1].iter().all(|l| l.len() > WIDTH - 5), "a line is cut early");
        assert_eq!(w.split_whitespace().count(), 60);
        assert_eq!(row("--a", "b"), format!("  --a{}b\n", " ".repeat(COLUMN - 5)));
        assert!(row(&"x".repeat(COLUMN), "b").contains(&format!("\n{}b", " ".repeat(COLUMN))));
    }
}
