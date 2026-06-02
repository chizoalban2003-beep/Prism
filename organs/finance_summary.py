"""Bundled organ: finance_summary — summarise transactions from a local ledger."""
ORGAN_META = {
    "intent": "finance_summary",
    "description": "Summarise recent transactions from a local CSV or JSON ledger file",
    "version": "1.0",
}


def execute(intent: str, message: str, ctx: dict):
    import json
    import re
    from collections import defaultdict
    from pathlib import Path

    from prism_responses import text_card

    default_path = str(Path.home() / ".prism" / "finance.json")
    ledger_path = Path(ctx.get("finance_ledger", default_path))

    if not ledger_path.exists():
        msg = (
            "No ledger file found.\n\n"
            f"Expected location: {ledger_path}\n\n"
            "To set up your ledger, create a JSON file containing an array of "
            "transaction objects:\n"
            '  [{"date": "2026-06-01", "amount": -12.50, '
            '"category": "Food", "description": "Lunch"}, ...]\n\n'
            "Positive amounts = income, negative amounts = expenses.\n"
            "Alternatively, use a CSV file with columns: "
            "date, amount, category, description.\n\n"
            "You can also pass a custom path via ctx['finance_ledger']."
        )
        return text_card(msg, intent)

    try:
        raw = ledger_path.read_text(encoding="utf-8")
        ext = ledger_path.suffix.lower()

        if ext == ".csv":
            import csv
            import io
            reader = csv.DictReader(io.StringIO(raw))
            transactions = [
                {
                    "date": row.get("date", ""),
                    "amount": float(row.get("amount", 0)),
                    "category": row.get("category", "Uncategorised"),
                    "description": row.get("description", ""),
                }
                for row in reader
            ]
        else:
            transactions = json.loads(raw)
            transactions = [
                {
                    "date": t.get("date", ""),
                    "amount": float(t.get("amount", 0)),
                    "category": t.get("category", "Uncategorised"),
                    "description": t.get("description", ""),
                }
                for t in transactions
            ]

        if not transactions:
            return text_card("Ledger file is empty.", intent)

        income = sum(t["amount"] for t in transactions if t["amount"] > 0)
        expenses = sum(t["amount"] for t in transactions if t["amount"] < 0)
        net = income + expenses

        category_totals: dict = defaultdict(float)
        for t in transactions:
            if t["amount"] < 0:
                category_totals[t["category"]] += abs(t["amount"])

        top3 = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:3]

        last5 = transactions[-5:]
        last5_lines = "\n".join(
            f"  {t['date']}  {t['category']:<16} "
            f"{'%+.2f' % t['amount']}  {t['description']}"
            for t in reversed(last5)
        )

        top3_lines = "\n".join(
            f"  {cat:<20} £{amt:,.2f}" for cat, amt in top3
        ) or "  (none)"

        result = (
            f"Finance Summary ({len(transactions)} transactions)\n"
            f"{'='*44}\n"
            f"  Total income:   £{income:>10,.2f}\n"
            f"  Total expenses: £{abs(expenses):>10,.2f}\n"
            f"  Net:            £{net:>10,.2f}\n\n"
            f"Top spending categories:\n{top3_lines}\n\n"
            f"Last 5 transactions:\n{last5_lines}"
        )
    except Exception as exc:
        result = f"Error reading ledger '{ledger_path}': {exc}"

    return text_card(result, intent)
