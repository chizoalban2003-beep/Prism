// PRISM VS Code Extension
// Bridges VS Code to the local PRISM assistant running on 127.0.0.1:8743.
// No npm dependencies — uses Node 18+ built-in fetch.

const vscode = require("vscode");

let outputChannel;
let statusBarItem;

function getBaseUrl() {
  const cfg = vscode.workspace.getConfiguration("prism");
  const host = cfg.get("host") || "127.0.0.1";
  const port = cfg.get("port") || 8743;
  return `http://${host}:${port}`;
}

async function prismFetch(path, method = "GET", body = null) {
  const url = getBaseUrl() + path;
  const options = { method, headers: { "Content-Type": "application/json" } };
  if (body) options.body = JSON.stringify(body);
  const res = await fetch(url, options);
  return res.json();
}

function showResult(title, text) {
  outputChannel.appendLine(`\n${"─".repeat(60)}`);
  outputChannel.appendLine(`[${new Date().toLocaleTimeString()}] ${title}`);
  outputChannel.appendLine("─".repeat(60));
  outputChannel.appendLine(text);
  outputChannel.show(true);
}

async function checkStatus() {
  try {
    const data = await prismFetch("/ide/status");
    const ready = data.ok && data.agent_ready;
    statusBarItem.text = ready ? "$(circle-filled) PRISM" : "$(circle-outline) PRISM";
    statusBarItem.color = ready
      ? new vscode.ThemeColor("statusBarItem.prominentForeground")
      : new vscode.ThemeColor("statusBarItem.errorForeground");
    statusBarItem.tooltip = ready
      ? `PRISM connected — phase: ${data.phase}`
      : "PRISM not ready — is prism_daemon running?";
  } catch {
    statusBarItem.text = "$(circle-slash) PRISM";
    statusBarItem.color = new vscode.ThemeColor("statusBarItem.errorForeground");
    statusBarItem.tooltip = "PRISM unreachable";
  }
}

async function explainCode() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("PRISM: No active editor.");
    return;
  }
  const selection = editor.selection;
  const code = editor.document.getText(selection.isEmpty ? undefined : selection).trim();
  if (!code) {
    vscode.window.showWarningMessage("PRISM: No code selected.");
    return;
  }
  const language = editor.document.languageId;
  const question = await vscode.window.showInputBox({
    prompt: "What would you like to know? (leave blank for general explanation)",
    placeHolder: "e.g. What does this function do?",
  });
  if (question === undefined) return; // cancelled
  const data = await prismFetch("/ide/explain", "POST", {
    code,
    language,
    question: question || undefined,
  });
  showResult("Explanation", data.explanation || data.error || JSON.stringify(data));
}

async function reviewCode() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("PRISM: No active editor.");
    return;
  }
  const code = editor.document.getText().trim();
  if (!code) {
    vscode.window.showWarningMessage("PRISM: File is empty.");
    return;
  }
  const language = editor.document.languageId;
  const filename = editor.document.fileName.split("/").pop();
  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "PRISM: Reviewing…" },
    async () => {
      const data = await prismFetch("/ide/review", "POST", { code, language, filename });
      showResult(`Code Review — ${filename}`, data.review || data.error || JSON.stringify(data));
    }
  );
}

async function fixError() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("PRISM: No active editor.");
    return;
  }
  const selection = editor.selection;
  const code = editor.document.getText(selection.isEmpty ? undefined : selection).trim();
  if (!code) {
    vscode.window.showWarningMessage("PRISM: No code selected.");
    return;
  }
  // Try to pick up diagnostic message from the current cursor position
  const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
  const cursorLine = editor.selection.active.line;
  const nearestDiag = diagnostics.find((d) => d.range.start.line === cursorLine);
  const defaultError = nearestDiag ? nearestDiag.message : "";
  const errorMessage = await vscode.window.showInputBox({
    prompt: "Paste the error message",
    value: defaultError,
  });
  if (!errorMessage) return;
  const language = editor.document.languageId;
  const data = await prismFetch("/ide/fix", "POST", { code, error_message: errorMessage, language });
  showResult("Suggested Fix", data.fix || data.error || JSON.stringify(data));
}

async function chat() {
  const editor = vscode.window.activeTextEditor;
  const message = await vscode.window.showInputBox({
    prompt: "Ask PRISM anything",
    placeHolder: "e.g. How do I optimise this loop?",
  });
  if (!message) return;
  let code_context;
  let filename;
  if (editor) {
    const sel = editor.selection;
    code_context = editor.document.getText(sel.isEmpty ? undefined : sel).trim() || undefined;
    filename = editor.document.fileName.split("/").pop();
  }
  const data = await prismFetch("/ide/chat", "POST", { message, code_context, filename });
  showResult("PRISM Chat", data.reply || data.error || JSON.stringify(data));
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel("PRISM");
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "prism.chat";
  statusBarItem.text = "$(circle-outline) PRISM";
  statusBarItem.show();

  checkStatus();
  const timer = setInterval(checkStatus, 30_000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
  context.subscriptions.push(statusBarItem, outputChannel);

  context.subscriptions.push(
    vscode.commands.registerCommand("prism.explainCode", explainCode),
    vscode.commands.registerCommand("prism.reviewCode", reviewCode),
    vscode.commands.registerCommand("prism.fixError", fixError),
    vscode.commands.registerCommand("prism.chat", chat)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
