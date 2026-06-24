"""
prism_intents.py
================
Static intent-routing table for PrismAgent: ordered (regex, intent) pairs.
First match wins (see PrismAgent._route). Extracted from prism_agent.py to
keep the agent module focused; behaviour is unchanged.
"""
from __future__ import annotations

INTENTS: list[tuple[str, str]] = [
    # Horizon goals — "when X" / "watch for" / "notify when" must precede every
    # other intent because the trigger clause ("when bitcoin drops") would
    # otherwise be consumed by topic keywords (bitcoin → web_search, etc).
    (r"watch (?:for|out for)|monitor (?:for|when)|track (?:when|until)|"
     r"(?:tell|alert|notify|remind|ping) me when|"
     r"(?:wait|keep watching) (?:for|until)|"
     r"(?:book|buy|do|send|run) (?:it |that )?when |"
     r"horizon goal|long.?term goal|background goal",
     "horizon_add"),
    # Live financial/crypto data — must precede plan ("today") and wikipedia ("what is")
    (r"stock (?:price|market|quote)|share price|market cap|"
     r"bitcoin|ethereum|crypto (?:price|market)|coin price|"
     r"(?:price|value) of (?:bitcoin|ethereum|[A-Z]{2,5})\b",
     "web_search"),
    # News must precede plan — "today's headlines" contains "today"
    (r"news|headlines|top stories|latest stories|breaking news", "news_headlines"),
    (r"(?!.*\bto (?:french|spanish|german|japanese|chinese|arabic|russian|hindi|italian"
     r"|portuguese)\b)(?:plan|morning|daily|today|schedule)", "universal_plan"),
    (r"how (?:do|can|should) i|plan (?:for|to)|strategy for|help me (?:with|plan)|"
     r"what(?:'s| is) the best way|i want to|i need to|my goal is", "universal_plan"),
    (r"predict|match|fixture|vs|versus", "predict_match"),
    (r"injury risk|squad risk|squad injury|player risk|player fitness|"
     r"\binjury\b|\bsquad\b|\bfitness\b", "squad_risk"),
    (r"moment|1v1|keeper|\bshot\b|attack", "moment"),
    (r"session|footage|video|analyse.*play", "session"),
    (r"transfer|market|value|worth", "transfer"),
    (r"triage|chest|pain|fever|symptom|patient", "domain_medical"),
    (r"portfolio|invest|allocation|bonds|equity", "domain_financial"),
    (r"legal|case|litigat|settle|arbitrat", "domain_legal"),
    (r"hire|hiring|recruit|talent|headcount", "domain_hr"),
    (r"supply chain|procurement|inventory|(?:stock|restock) (?:level|order|management)|out of stock",
     "domain_supply"),
    (r"climate|carbon|emission|energy\.policy", "domain_climate"),
    (r"what (?:do you )?know about me|my profile|who am i|crystallise|persona|how well do you know me",
     "my_profile"),
    (r"my (?:week|weekly|month|monthly) (?:report|summary|narrative|review)|"
     r"what happened this (?:week|month)",
     "my_narrative"),
    (r"how (?:much have you |have you )learned|growth report|"
     r"what have you learned about me|prism growth",
     "my_growth"),
    (r"identity|digital\.dna|who\.am", "identity"),
    (r"artifact|past\.decision|what\.have\.i|my artifacts", "artifacts"),
    (r"\bstatus\b|connected|device|\bsync\b", "status"),
    # Personal-fact recall — "what is my favourite colour", "do you remember
    # my partner's name". Placed after the specific my_X intents (profile,
    # narrative, growth, identity, artifacts, status) and before the generic
    # wikipedia_lookup/web_search catch-alls, so retrievable personal facts
    # don't get redirected to an encyclopaedia. The negative lookahead lists
    # tokens that have their own dedicated route.
    (r"(?:what(?:'s| is| are)|tell me|do you (?:know|remember)|recall|"
     r"what did i (?:say|tell you) about)\s+(?:about\s+)?my\b"
     r"(?!\s+(?:profile|narrative|growth|week|month|tasks?|todos?|to-?do|"
     r"budget|spend|polic|limit|instructions?|rules?|standing|horizon|"
     r"organs?|feedback|inbox|mailbox|email|mail|messages?|calendar|"
     r"schedule|agenda|meetings?|appointments?|events?|files?|downloads?|"
     r"documents?|desktop|pictures?|music|videos?|finances?|transactions?|"
     r"expenses?|health|steps?|sleep|hrv|heart|calories?|artifacts?|"
     r"identity|persona|status|clipboard|contacts?|day|mind|screen))",
     "memory_recall"),
    (r"index|scan\.files|search\.code|grep|find\.file", "ksa_task"),
    (r"resize|(?:convert|compress) (?:file|image|video)|rename|move|copy|delete|create file|"
     r"find file|search (?:in|for)|read file|list files|"
     r"run (?:command|script)|execute|open (?:app|file)|"
     r"install (?:package|app)|git (?:commit|push|pull|status)|"
     r"what(?:'s| is) (?:on|in) my(?! screen| calendar| schedule| agenda| inbox| email| mailbox)|"
     r"show me (?:my )?files|"
     r"\bmy files?\b|"
     r"\bmy (?:downloads|documents|desktop|pictures|music|videos)\b",
     "device_task"),
    (r"show (?:my )?polic|what(?:'s| are) my (?:budget|polic|limit)|"
     r"current (?:polic|budget|limit)", "show_policies"),
    (r"set (?:my )?(\w+) (?:budget|limit)|auto.?approv|never use|"
     r"require approval|reset (?:all )?polic", "update_policy"),
    (r"(?:running|active|pending|recent) tasks?|task (?:status|progress)|"
     r"what(?:'s| is) (?:running|happening)", "task_status"),
    (r"(?:read|check|show|any|my) (?:new )?(?:emails?|inbox|messages?)|"
     r"unread|what(?:'s| came) in",                        "email_read"),
    (r"(?:send|reply|write|draft) (?:an? )?email|"
     r"email (?:to|them|him|her)",                          "email_send"),
    (r"(?:what(?:'s| is) on my|check my|show) (?:calendar|schedule|agenda)|"
     r"(?:any|my) (?:meetings?|appointments?|events?) (?:today|tomorrow|this week)",
                                                            "calendar_read"),
    (r"(?:schedule|book|create|add) (?:a )?(?:meeting|event|appointment)|"
     r"(?:find|when(?:'s| is) the next) (?:free|available) (?:slot|time)",
                                                            "calendar_write"),
    (r"(?:look up|tell me about|what (?:is|was|are)) .+?(?:on |in )?wikipedia|wikipedia",
     "wikipedia_lookup"),
    (r"(?:go to|open|browse|visit|find (?:on|online)|"
     r"look up(?! .+ (?:on|in) wikipedia)|book|reserve|fill (?:in|out)|"
     r"check (?:the )?(?:price|availability)|what(?:'s| is) (?:on|the) website)",
     "browser_task"),
    (r"show (?:my )?(?:instructions?|rules?|standing orders?)|"
     r"what (?:have you )?(?:remember|know) about my preferences",
     "show_instructions"),
    (r"(?:forget|remove|delete) (?:that |the )?(?:instruction|rule)|"
     r"stop (?:always|never)", "remove_instruction"),
    (r"(?:use|connect|integrate|set up|configure|add) (?:with )?(?!my )(?!the )"
     r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[a-z]+\.[a-z]+)|"
     r"(?:can you|how do i) (?:use|access|connect to) ", "discover_service"),
    (r"search (?:the web|online|internet|for)|"
     r"look up|find (?:out|info|information)|"
     r"what(?:'s| is) (?:the )?(?:latest|current|today)|"
     r"research|who is|where is|when (?:did|does|is)",
     "web_search"),
    (r"(?:send|push) (?:me )?(?:a )?(?:notification|alert|reminder)|"
     r"notify me|ping me|alert me",
     "send_push"),
    (r"(?:find|search|look up|who is|contact|call|email) (?:my )?(?:contact|person|colleague|client|friend)",
     "contacts"),
    (r"(?:append|add|write|save|take|jot down) (?:a )?note|note(?:pad)?:? ", "note_append"),
    (r"remind me|set (?:a )?reminder|alert me (?:in|at|when)|"
     r"don't let me forget|in (\d+) (?:minute|hour|day)|"
     r"at (\d+(?::\d+)?(?:am|pm)?)", "reminder_set"),
    (r"(?:add|create|make|new) (?:a )?(?:task|todo|reminder|ticket|issue)|"
     r"(?:i need to|i have to|remember to|don't forget)",
     "add_task"),
    (r"(?:my )?(?:tasks?|todos?|to-do|to do|what(?:'s| is) (?:on my )?list|"
     r"pending|backlog|open issues?)",
     "list_tasks"),
    (r"(?:that was|you were|that(?:'s| is)) (?:wrong|right|too|not|off|correct|"
     r"perfect|bad|good)|(?:i (?:disagree|agree|wouldn't|would)|"
     r"too (?:aggressive|cautious|risky|safe|bold|timid)|"
     r"next time (?:consider|weight|prioritise)|"
     r"(?:more|less) (?:important|weight|focus))", "calibrate"),
    (r"(?:how am i|calibration|what have you learned|"
     r"how (?:accurate|well) (?:are you|is prism)|"
     r"show (?:my )?feedback history)", "calibration_summary"),
    (r"^(?:yes[,.]?|yeah[,.]?|go ahead|approved?|confirm|do it|proceed)[\s!.]*$",
     "approve_pending"),
    (r"^(?:no[,.]?|cancel|stop|don't|abort|never mind)[\s!.]*$",
     "cancel_pending"),
    (r"(?:what tools|learned tools|acquired tools|"
     r"what can you now do|new capabilities|tool list)",
     "list_tools"),
    (r"how did you do that|what steps did you take|"
     r"explain (?:your )?(?:steps?|process|plan)|"
     r"show (?:me )?(?:the )?steps",
     "explain_composition"),
    (r"show (?:me )?(?:the )?chain|chain (?:steps?|history|log)|"
     r"how did (?:the )?chain work|what (?:steps?|logics?) did you use",
     "chain_history"),
    (r"outcome stats?|chain outcomes?|learning stats?|"
     r"completion rate|how (?:often|many chains?) (?:do you )?(?:complete|finish)",
     "outcome_stats"),
    (r"\b(?:show|what(?:'s| is)?|check)\s+(?:my\s+|the\s+)?budget\b|"
     r"\b(?:llm|api|prism)\s+(?:spend|cost|budget|spending)\b|"
     r"\bhow much (?:have (?:i|you)|did (?:i|you)) spen[dt]\b|"
     r"\bdaily (?:cost|spend|budget)\b",
     "budget_status"),
    (r"weekly reflection|reflect (?:on (?:this|the|my) )?(?:week|month|today)|"
     r"how (?:did|was) (?:my|the|this) (?:week|month) go|"
     r"weekly summary",
     "reflection"),
    (r"(?:start|begin|enable|use) (?:voice|microphone|listening|speech)|"
     r"(?:stop|disable) (?:voice|listening|speech)|"
     r"(?:voice|speech|microphone) (?:on|off|status|available)|"
     r"(?:transcribe|listen|record) (?:audio|voice|speech|this)",
     "voice"),
    (r"help|what\.can|commands|options", "help"),
    # Horizon Planner — cross-session goal watching.
    # horizon_add itself is hoisted to the top of this list so trigger
    # clauses like "notify me when bitcoin drops" win against topic
    # keyword routes. list/abandon don't need hoisting because their
    # phrasing doesn't collide with other intents.
    (r"(?:show|list|what are) (?:my )?(?:horizon|background|watching|monitored) goals?|"
     r"what (?:are you |is prism )?(?:watching|monitoring|tracking)|"
     r"horizon (?:status|goals?|list)",
     "horizon_list"),
    (r"(?:stop|cancel|abandon) (?:watching|monitoring|tracking|that horizon|horizon goal)|"
     r"(?:forget|remove|delete) (?:that )?(?:goal|watch|monitor)",
     "horizon_abandon"),
    # Organs — loaded capabilities
    (r"(?:what|which|show|list) (?:my )?(?:organs?|loaded (?:capabilities|modules|tools))|"
     r"organ (?:list|status|registry)",
     "list_organs"),
    (r"turn (?:on|off)|set (?:the )?(?:lights?|thermostat|temp)|"
     r"\b(?:un)?lock\b|what(?:'s| is) (?:on|off)(?! (?:my|the) screen)|smart home|home assistant",
     "smart_home"),
    # NOTE: broad email catch-all — maps to email_read to avoid duplication
    # with the more specific email_read/email_send intents above.
    (r"(?:check|read|show|open|fetch|get|list).*(?:email|inbox|mail)|"
     r"(?:email|mail).*(?:unread|new|recent)|send.*(?:email|mail)|"
     r"draft.*(?:email|reply)|reply.*email|email.*summary",
     "email_read"),
    # Organ-mapped intents (broad fallback patterns — do not duplicate entries above)
    (r"weather|temperature|forecast|how (?:hot|cold)|rain|sunny", "weather_check"),
    (r"wikipedia|look up|tell me about|who (?:is|was)|what (?:is|was) (?:a |an |the )?[A-Za-z]",
     "wikipedia_lookup"),
    (r"translate|translation|in (?:spanish|french|german|italian|portuguese|chinese|japanese|arabic|russian|hindi)",
     "translate_text"),
    # Physical-unit disambiguation FIRST: when an unambiguous metric/imperial
    # unit token is present (kg, miles, celsius, …) treat it as a unit
    # conversion even if a currency word like "pounds" also appears
    # ("10 kg to pounds" is weight, not GBP). Currency words (usd, dollar…)
    # are deliberately excluded here so real FX requests fall through.
    (r"\b(?:km|kilometers?|kilometres?|miles?|kg|kilograms?|grams?|mg|"
     r"ounces?|oz|stones?|celsius|fahrenheit|kelvin|meters?|metres?|cm|mm|"
     r"feet|foot|inch|inches|yards?|liters?|litres?|ml|gallons?|pints?|"
     r"mph|kph|km/h)\b[^.]{0,30}?\b(?:to|in|into)\b",
     "unit_convert"),
    (r"(?:convert|exchange|how much) .* (?:usd|eur|gbp|jpy|cad|aud|chf|cny|currency)|"
     r"(?:usd|eur|gbp|jpy|cad|aud|chf|cny) (?:to|in|into)|"
     r"(?:dollar|euro|pound|yen|yuan|franc|rupee|peso|won|ruble|lira|krona|"
     r"baht|ringgit|dirham|real|shekel|zloty|forint|koruna|krone|dinar|"
     r"bitcoin|satoshi|ethereum) (?:to|in|into)|"
     r"(?:convert|exchange) .* (?:dollar|euro|pound|yen|yuan|franc|rupee)",
     "currency_convert"),
    (r"(?:convert|how many|how much) .* (?:to|in|into)|"
     r"(?:km|miles|kg|lbs|celsius|fahrenheit|meters?|feet|inches?|liters?|gallons?) (?:to|in|into)",
     "unit_convert"),
    (r"(?:take|capture|grab) (?:a )?screenshot|screenshot", "screenshot_capture"),
    (r"what(?:'s| is) on (?:my |the )?screen|analyse (?:my |the )?screen|"
     r"analyze (?:my |the )?screen|describe (?:my |the )?screen|"
     r"look at (?:my |the )?screen|what do you see|vision query|"
     r"read (?:my |the )?screen|what(?:'s| is) (?:happening|visible) on screen",
     "vision_query"),
    (r"(?:read|what(?:'s| is) on|show|paste|get) (?:my )?clipboard", "clipboard_read"),
    (r"(?:set|start|create) (?:a )?timer|timer (?:for|of)|countdown", "timer_set"),
    (r"(?:read|open|show|cat|display) (?:my |the )?file|file (?:contents?|read)", "file_read"),
    (r"(?:write|save|create|overwrite) (?:to )?(?:the )?file|write (?:this|that) to", "file_write"),
    (r"(?:play|pause|skip|next|previous|volume|stop) (?:music|spotify|song|track|playback)",
     "spotify_control"),
    (r"(?:generate|create|make|qr) (?:a )?qr (?:code)?|qr code for", "qr_generate"),
    (r"(?:run|execute|shell|bash|cmd|terminal|command)(?:\s|:)", "shell_run"),
    (r"(?:make|place|give|dial) (?:a )?(?:phone )?call|"
     r"(?:call|phone|ring) (?:someone|them|him|her|my |the )|"
     r"phone call to|\bcall \d",
     "phone_call"),
    (r"github (?:issue|pr|pull request|repo)|(?:create|list|open) (?:an? )?issue", "github_issue"),
    (r"(?:send|post) (?:a )?(?:message )?(?:to|on) discord|discord", "discord_send"),
    (r"(?:send|post) (?:a )?(?:message )?(?:to|on) telegram|telegram", "telegram_send"),
    (r"(?:control|turn|set|dim) (?:the )?(?:lights?|thermostat|fan|ac|heater|lock|switch)",
     "smart_home_control"),
    (r"my (?:finances?|budget|spending|transactions?|expenses?)|finance (?:summary|report)",
     "finance_summary"),
    (r"my (?:health|steps?|sleep|hrv|heart rate|calories?)|health (?:summary|report|data)",
     "health_summary"),
    (r"(?:brief|briefing|prep|summary) (?:for|before|about) (?:my )?(?:meeting|call|standup)",
     "meeting_brief"),
    (r"(?:overdue|due today|pending|upcoming) (?:tasks?|reminders?|todos?)|task reminder",
     "task_reminder"),
    (r"policy (?:audit|log|history)|audit (?:log|trail)", "policy_audit"),
    # NOTE: entries below are intentionally absent — the following patterns
    # were duplicates of earlier INTENTS entries and have been removed:
    #   news_headlines (subset of the first news entry above)
]
